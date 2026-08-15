"""Audio Service - streams streamline-generated samples to the sound device.

    tracer compute shader -> SSBO slot ring -> fence -> read_into
        -> CPU limiter -> rtmixer lock-free ring -> PortAudio callback

Single-threaded: blocks are produced inside the render loop. The GPU slot ring
absorbs GPU-side jitter; the rtmixer ring absorbs render-loop jitter.

Structure follows demos/audio.py. The difference is that the samples are not
synthesised here - the streamline tracer writes them, one per integration
step, and this service only moves and conditions them.
"""
import ctypes
import math
import sys

import numpy as np

from state.audio_state import AUDIO_BLOCK, AUDIO_SLOTS, AUDIO_RB_BLOCKS

# GL sync constants. moderngl wraps none of these, so they come from the
# driver directly.
GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
GL_ALREADY_SIGNALED = 0x911A
GL_CONDITION_SATISFIED = 0x911C
GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001
GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000
GL_BUFFER_UPDATE_BARRIER_BIT = 0x00000200

# GL uses __stdcall on Windows.
_FUNC = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE

AUDIO_BINDING = 7


def _gl(name, restype, *argtypes):
    import glfw
    addr = glfw.get_proc_address(name)
    if not addr:
        raise RuntimeError(f"Driver did not export {name} (need GL 4.3+)")
    if not isinstance(addr, int):
        addr = ctypes.cast(addr, ctypes.c_void_p).value
    return _FUNC(restype, *argtypes)(addr)


class GLSync:
    """Fence helpers, loaded after the context exists."""

    def __init__(self):
        self.fence_sync = _gl("glFenceSync", ctypes.c_void_p,
                              ctypes.c_uint, ctypes.c_uint)
        self.client_wait = _gl("glClientWaitSync", ctypes.c_uint,
                               ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64)
        self.delete_sync = _gl("glDeleteSync", None, ctypes.c_void_p)
        self.memory_barrier = _gl("glMemoryBarrier", None, ctypes.c_uint)

    def fence(self):
        return self.fence_sync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    def signalled(self, sync):
        """Non-blocking poll. FLUSH_COMMANDS_BIT avoids waiting forever on
        commands still sitting unflushed in the driver queue."""
        r = self.client_wait(sync, GL_SYNC_FLUSH_COMMANDS_BIT, 0)
        return r in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED)

    def barrier(self):
        self.memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT
                            | GL_BUFFER_UPDATE_BARRIER_BIT)


class Limiter:
    """Block-rate peak limiter: instant attack, exponential release, gain
    ramped across the block to avoid zipper noise. A hard clip backstops it."""

    def __init__(self, sample_rate, ceiling=0.9, release_ms=120.0):
        self.ceiling = ceiling
        self.gain = 1.0
        self.release = math.exp(-1.0 / (release_ms * 0.001 * sample_rate / AUDIO_BLOCK))
        self.reduction_db = 0.0

    def process(self, block):
        peak = float(np.max(np.abs(block))) or 1e-9
        desired = min(1.0, self.ceiling / peak)
        if desired < self.gain:
            target = desired                     # attack: instant
        else:
            target = desired + (self.gain - desired) * self.release
        ramp = np.linspace(self.gain, target, len(block), dtype="f4")
        block *= ramp
        np.clip(block, -1.0, 1.0, out=block)
        self.gain = target
        self.reduction_db = 20.0 * math.log10(max(target, 1e-6))


class AudioService:
    """Owns the GPU audio ring, the CPU ring, and the output stream.

    The tracer writes one float per integration step into a slot of the GPU
    ring; this service fences each dispatch, reads back completed slots
    without stalling, conditions them, and hands them to rtmixer.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.sync = None
        self.buffer = None
        self.fences = [None] * AUDIO_SLOTS
        self.w = 0                # Next slot to dispatch into
        self.r = 0                # Next slot to read back
        self.staging = np.zeros(AUDIO_BLOCK, dtype="f4")
        self.stereo = np.zeros((AUDIO_BLOCK, 2), dtype="f4")
        self.limiter = None
        self.mixer = None
        self.rb = None
        self._action = None
        self.active = False
        self.starves = 0
        self.peak = 0.0
        self.device_name = ""
        self.last_error = ""
        self._sample_rate = 0
        # Running peak estimate for auto-gain. dot(vel, field) is unnormalised
        # and routinely reaches the hundreds, so without this the Amplitude
        # slider would need to live down at ~0.003.
        self._envelope = 1.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _ensure_gpu(self):
        if self.sync is None:
            self.sync = GLSync()
        if self.buffer is None:
            self.buffer = self.ctx.buffer(
                np.zeros(AUDIO_SLOTS * AUDIO_BLOCK, dtype="f4").tobytes()
            )

    def start(self, settings) -> bool:
        """Open the output stream. Returns True on success."""
        if self.active:
            return True
        try:
            import rtmixer
        except Exception as e:
            self.last_error = f"rtmixer unavailable: {e}"
            return False

        try:
            self._ensure_gpu()
            sr = int(settings.sample_rate)
            self._sample_rate = sr
            self.limiter = Limiter(sr, ceiling=settings.limiter_ceiling)

            rb_frames = AUDIO_BLOCK * AUDIO_RB_BLOCKS
            assert rb_frames & (rb_frames - 1) == 0, "ring size must be a power of 2"
            # 4 bytes per float * 2 channels per frame.
            self.rb = rtmixer.RingBuffer(4 * 2, rb_frames)
            self.mixer = rtmixer.Mixer(channels=2, samplerate=sr,
                                       blocksize=256, latency="low")
            self.mixer.start()
            # Playback starts in ensure_playing() once the ring has been
            # prefilled. Starting it against an empty ring would underrun
            # immediately, and rtmixer retires the action permanently on
            # underrun rather than waiting for more data.
            self._action = None

            try:
                import sounddevice as sd
                self.device_name = sd.query_devices(kind="output")["name"]
            except Exception:
                self.device_name = "default"

            self.w = self.r = 0
            self.fences = [None] * AUDIO_SLOTS
            self.starves = 0
            self.active = True
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = f"audio start failed: {e}"
            self.stop()
            return False

    def stop(self):
        """Close the output stream, leaving GPU resources allocated."""
        if self.mixer is not None:
            try:
                self.mixer.stop()
            except Exception:
                pass
        self.mixer = None
        self.rb = None
        self._action = None
        self.active = False
        for i, f in enumerate(self.fences):
            if f is not None:
                self.sync.delete_sync(f)
                self.fences[i] = None
        self.w = self.r = 0

    def cleanup(self):
        self.stop()
        if self.buffer is not None:
            self.buffer.release()
            self.buffer = None

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------
    def bind(self):
        """Bind the audio SSBO so the tracer can write into it."""
        self._ensure_gpu()
        self.buffer.bind_to_storage_buffer(AUDIO_BINDING)

    @property
    def slot_base(self) -> int:
        """Sample index where the next dispatch should write its block."""
        return (self.w % AUDIO_SLOTS) * AUDIO_BLOCK

    def can_dispatch(self) -> bool:
        """True while there is a free GPU slot.

        Keeping one slot spare means a dispatch never overwrites a block that
        has been produced but not yet read back.
        """
        return (self.w - self.r) < (AUDIO_SLOTS - 1)

    def note_dispatch(self):
        """Fence the dispatch that was just issued and claim its slot."""
        slot = self.w % AUDIO_SLOTS
        self.sync.barrier()
        self.fences[slot] = self.sync.fence()
        self.w += 1

    def blocks_wanted(self) -> int:
        """How many blocks the CPU ring currently has room for."""
        if not self.active:
            return 0
        return max(0, self.rb.write_available // AUDIO_BLOCK)

    def pump(self, settings) -> int:
        """Move every ready GPU block into the audio ring. Returns count."""
        if not self.active:
            return 0
        drained = 0
        while self.rb.write_available >= AUDIO_BLOCK and self.r < self.w:
            slot = self.r % AUDIO_SLOTS
            fence = self.fences[slot]
            if fence is None or not self.sync.signalled(fence):
                break
            # The fence has signalled, so this is a plain memcpy, not a stall.
            self.buffer.read_into(self.staging, size=AUDIO_BLOCK * 4,
                                  offset=slot * AUDIO_BLOCK * 4)
            self.sync.delete_sync(fence)
            self.fences[slot] = None
            self.r += 1

            block = self.staging
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            if settings.auto_gain:
                # Normalise by a tracked peak. The raw signal's scale depends
                # entirely on how much energy the canvas holds - it spans
                # several orders of magnitude between an empty canvas and a
                # dense one - so this is what keeps Amplitude meaningful.
                blk_peak = float(np.max(np.abs(block)))
                if blk_peak > self._envelope:
                    self._envelope = blk_peak            # attack: instant
                else:
                    # Release in the log domain. The raw level tracks canvas
                    # energy and spans decades, so a linear release crawls
                    # down from a transient and leaves everything after it
                    # inaudible. Measured range is ~27 dB within a run and
                    # far wider between an empty and a dense canvas.
                    ratio = max(blk_peak, 1e-9) / self._envelope
                    self._envelope *= ratio ** 0.25
                # Floor the envelope well below any usable signal, so silence
                # is not amplified into noise.
                self._envelope = max(self._envelope, 1e-6)
                gain = float(settings.amplitude) / self._envelope
                # Cap the boost so a near-silent canvas cannot scream when it
                # suddenly gains energy.
                gain = min(gain, 1e4)
                block *= gain
                settings.auto_gain_db = 20.0 * math.log10(max(gain, 1e-9))
            else:
                self._envelope = 1.0
                settings.auto_gain_db = 0.0

            self.limiter.ceiling = float(settings.limiter_ceiling)
            self.limiter.process(block)
            # Report the peak of what actually leaves the pipeline, not the
            # limiter's input: the input peak is pre-gain and so looks
            # identical however the Amplitude slider is set.
            self.peak = float(np.max(np.abs(block)))
            # Mono voice -> both channels.
            self.stereo[:, 0] = block
            self.stereo[:, 1] = block
            self.rb.write(self.stereo)
            drained += 1

        self.ensure_playing()

        if self.rb.read_available < AUDIO_BLOCK:
            self.starves += 1
        return drained

    def ensure_playing(self):
        """(Re)start playback once the ring holds enough to survive a hitch.

        rtmixer retires a play_ringbuffer action permanently when the ring
        runs dry, so this both performs the initial prefill-then-start and
        recovers from any later underrun.
        """
        if not self.active or self.rb is None:
            return
        if self._action is not None and self._action in self.mixer.actions:
            return  # Still playing.
        # Wait for a healthy buffer before (re)starting, otherwise playback
        # dies again on the very next callback.
        if self.rb.read_available >= AUDIO_BLOCK * (AUDIO_RB_BLOCKS // 2):
            try:
                self._action = self.mixer.play_ringbuffer(self.rb)
            except Exception as e:
                self.last_error = f"playback restart failed: {e}"

    def update_telemetry(self, settings):
        settings.starves = self.starves
        settings.in_flight = self.w - self.r
        settings.peak = float(self.peak)
        settings.device_name = self.device_name
        settings.last_error = self.last_error
        if self.active and self.rb is not None:
            total = AUDIO_BLOCK * AUDIO_RB_BLOCKS
            settings.ring_fill = self.rb.read_available / total
        else:
            settings.ring_fill = 0.0

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------
    def highpass_coeff(self, settings) -> float:
        """One-pole high-pass coefficient for the configured cutoff."""
        sr = float(settings.sample_rate)
        fc = max(1.0, float(settings.highpass_hz))
        # Standard DC-blocker pole: R = 1 - 2*pi*fc/sr, clamped for stability.
        return float(min(0.99999, max(0.0, 1.0 - (2.0 * math.pi * fc / sr))))

    def ramp_decrement(self, settings) -> float:
        """Per-sample decrement so the ramp spans reset_ramp_ms."""
        n = max(1.0, float(settings.reset_ramp_ms) * 0.001 * settings.sample_rate)
        return float(1.0 / n)
