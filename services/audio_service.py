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
from state.streamline_state import MAX_STREAMLINES
from utilities.gl_helpers import read_shader, tryset

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

AUDIO_BINDING = 7   # Per-voice lanes
MIX_BINDING = 8     # Reduced mono mix


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
        self.buffer = None       # Per-voice lanes, written by the tracer
        self.mix_buffer = None   # Reduced mono mix, read back to the CPU
        self.mix_program = None
        self.fences = [None] * AUDIO_SLOTS
        self.w = 0                # Next slot to dispatch into
        self.r = 0                # Next slot to read back
        # Samples already written into the slot at w. The interleaved producer
        # fills a block across several physics intervals rather than in one
        # dispatch, so a block is only complete - reduced and fenced - once
        # this reaches AUDIO_BLOCK.
        self.block_fill = 0
        self.staging = np.zeros(AUDIO_BLOCK, dtype="f4")
        self.stereo = np.zeros((AUDIO_BLOCK, 2), dtype="f4")
        self.limiter = None
        self.mixer = None
        self.rb = None
        self._action = None
        self.active = False
        self.starves = 0
        # Blocks produced but dropped because the CPU ring was full.
        self.overruns = 0
        # Fractional blocks owed to the device; see blocks_wanted().
        self._block_debt = 0.0
        # Fences created minus fences deleted. Should oscillate in [0, SLOTS]
        # and never trend upward; a climbing value means sync objects are
        # leaking into the driver.
        self.live_fences = 0
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
            # One lane of (AUDIO_SLOTS * AUDIO_BLOCK) floats per possible
            # voice. 1024 * 16 * 512 * 4B = 33.6 MB.
            self.buffer = self.ctx.buffer(
                np.zeros(MAX_STREAMLINES * self.lane_stride, dtype="f4").tobytes()
            )
        if self.mix_buffer is None:
            self.mix_buffer = self.ctx.buffer(
                np.zeros(AUDIO_SLOTS * AUDIO_BLOCK, dtype="f4").tobytes()
            )
        if self.mix_program is None:
            try:
                self.mix_program = self.ctx.compute_shader(
                    read_shader('shaders/streamline_audio_mix.glsl')
                )
            except Exception as e:
                print('Audio mix shader compilation failed:')
                print(e)
                self.last_error = f"mix shader failed: {e}"

    @property
    def lane_stride(self) -> int:
        """Floats per voice lane."""
        return AUDIO_SLOTS * AUDIO_BLOCK

    def reload(self):
        """Recompile the mix shader. Called on the V-key shader reload."""
        try:
            prog = self.ctx.compute_shader(
                read_shader('shaders/streamline_audio_mix.glsl')
            )
        except Exception as e:
            print('Audio mix shader compilation failed:')
            print(e)
            return
        if self.mix_program is not None:
            self.mix_program.release()
        self.mix_program = prog
        print('Audio mix shader reloaded')

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

            # Any fence still held from a previous run has to be deleted, not
            # dropped on the floor by reassigning the list.
            self._release_fences()
            self.w = self.r = 0
            self.block_fill = 0
            self.starves = 0
            self.overruns = 0
            self._block_debt = 0.0
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
        self._release_fences()
        self.w = self.r = 0
        self.block_fill = 0

    def _release_fences(self):
        """Delete every outstanding fence.

        Guarded on self.sync: start() can fail before _ensure_gpu() has run,
        and stop() is called from that failure path.
        """
        if self.sync is None:
            return
        for i, f in enumerate(self.fences):
            if f is not None:
                self.sync.delete_sync(f)
                self.live_fences -= 1
                self.fences[i] = None

    def cleanup(self):
        self.stop()
        for name in ('buffer', 'mix_buffer', 'mix_program'):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
                setattr(self, name, None)

    # ------------------------------------------------------------------
    # Production
    # ------------------------------------------------------------------
    def bind(self):
        """Bind the audio SSBOs so the tracer can write into them."""
        self._ensure_gpu()
        self.buffer.bind_to_storage_buffer(AUDIO_BINDING)
        self.mix_buffer.bind_to_storage_buffer(MIX_BINDING)

    @property
    def slot_base(self) -> int:
        """Sample index where the next dispatch should write its block."""
        return (self.w % AUDIO_SLOTS) * AUDIO_BLOCK

    def reduce(self, settings, voice_count: int):
        """Sum the per-voice lanes for the block just dispatched.

        Runs before the fence, so the fence covers both the tracer's writes
        and the reduction; the readback then sees a finished mix.
        """
        if self.mix_program is None:
            return
        n = max(1, int(voice_count))
        gain = 1.0 / math.sqrt(n) if settings.rms_normalise else 1.0
        tryset(self.mix_program, 'VOICE_COUNT', n)
        tryset(self.mix_program, 'LANE_STRIDE', self.lane_stride)
        tryset(self.mix_program, 'SLOT_BASE', self.slot_base)
        tryset(self.mix_program, 'BLOCK_SIZE', AUDIO_BLOCK)
        tryset(self.mix_program, 'MIX_GAIN', float(gain))
        self.mix_program.run((AUDIO_BLOCK + 63) // 64, 1, 1)

    def advance_fill(self, samples: int) -> bool:
        """Record `samples` written into the block in flight.

        Returns True when that block is now complete, which is the caller's
        signal to reduce and fence it.
        """
        self.block_fill += int(samples)
        if self.block_fill >= AUDIO_BLOCK:
            self.block_fill = 0
            return True
        return False

    def can_dispatch(self) -> bool:
        """True while there is a free GPU slot.

        Keeping one slot spare means a dispatch never overwrites a block that
        has been produced but not yet read back.
        """
        return (self.w - self.r) < (AUDIO_SLOTS - 1)

    def note_dispatch(self):
        """Fence the dispatch that was just issued and claim its slot."""
        slot = self.w % AUDIO_SLOTS
        # The tracer already issued a shader-storage barrier after its last
        # sub-dispatch and after the reduction, so the ordering this fence
        # needs is established; no extra barrier here.
        stale = self.fences[slot]
        if stale is not None:
            # Should not happen while can_dispatch() is respected, but never
            # overwrite a live GLsync handle - that is an unrecoverable leak.
            self.sync.delete_sync(stale)
            self.live_fences -= 1
        self.fences[slot] = self.sync.fence()
        self.live_fences += 1
        self.w += 1

    def blocks_wanted(self, dt: float = 0.0) -> int:
        """How many blocks to produce now.

        Paced by what the sound device actually consumes, not by how much room
        the ring happens to have. Ring space alone is a free-running producer:
        the readback frees slots in the same frame, so the tracer refills space
        that is about to be consumed and the surplus is discarded (measured
        2.14 blocks/frame produced against 1.57 consumed - a 36% overrun).
        That waste is cheap while a step is one texture read and expensive once
        each step is a full particle evaluation.

        The debt is accumulated in fractional blocks so the long-run rate is
        exact, then bounded by ring space and free GPU slots.
        """
        if not self.active:
            return 0

        room = max(0, self.rb.write_available // AUDIO_BLOCK)

        # Prefill: get the ring to a working depth before pacing takes over,
        # otherwise playback starts against a nearly empty buffer.
        target = AUDIO_RB_BLOCKS // 2
        if self.rb.read_available // AUDIO_BLOCK < target:
            self._block_debt = 0.0
            return room

        self._block_debt += max(0.0, dt) * self._sample_rate / AUDIO_BLOCK
        # Never bank more than a ring's worth; a long hitch should resume at
        # the current time, not replay the backlog.
        self._block_debt = min(self._block_debt, float(AUDIO_RB_BLOCKS))
        n = int(self._block_debt)
        n = min(n, room)
        self._block_debt -= n
        return n

    def pump(self, settings) -> int:
        """Move every ready GPU block into the audio ring. Returns count."""
        if not self.active:
            return 0
        drained = 0
        # Retire every signalled slot, whether or not there is ring space for
        # its samples. Gating this on write_available (as it once was) leaked
        # the fence whenever the CPU ring was full: the slot stayed claimed and
        # its GLsync was never deleted, so the driver's fence table grew without
        # bound for as long as audio ran. Dropping a block is a click; leaking
        # sync objects degrades the whole display driver.
        while self.r < self.w:
            slot = self.r % AUDIO_SLOTS
            fence = self.fences[slot]
            if fence is None or not self.sync.signalled(fence):
                break

            have_room = self.rb.write_available >= AUDIO_BLOCK
            if have_room:
                # The fence has signalled, so this is a plain memcpy, not a
                # stall. Read the reduced mix, not the per-voice lanes.
                self.mix_buffer.read_into(self.staging, size=AUDIO_BLOCK * 4,
                                          offset=slot * AUDIO_BLOCK * 4)

            self.sync.delete_sync(fence)
            self.live_fences -= 1
            self.fences[slot] = None
            self.r += 1

            if not have_room:
                # Ring is full: the slot is reclaimed but the samples are
                # discarded. The producer is ahead of the sound device, so
                # this is overrun, not starvation.
                self.overruns += 1
                continue

            # Same conditioning the offline render uses, so a recorded
            # file matches what the preview sounded like.
            block = self.condition_block(settings, self.staging)

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

    def condition_block(self, settings, block: np.ndarray) -> np.ndarray:
        """Apply the same auto-gain and limiting the realtime path uses.

        Shared so an offline render sounds like the preview did, rather than
        re-implementing the conditioning and drifting away from it.
        """
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if settings.auto_gain:
            blk_peak = float(np.max(np.abs(block)))
            if blk_peak > self._envelope:
                self._envelope = blk_peak
            else:
                ratio = max(blk_peak, 1e-9) / self._envelope
                self._envelope *= ratio ** 0.25
            self._envelope = max(self._envelope, 1e-6)
            gain = min(float(settings.amplitude) / self._envelope, 1e4)
            block *= gain
            settings.auto_gain_db = 20.0 * math.log10(max(gain, 1e-9))
        else:
            self._envelope = 1.0
            settings.auto_gain_db = 0.0

        if self.limiter is None:
            self.limiter = Limiter(int(settings.sample_rate),
                                   ceiling=settings.limiter_ceiling)
        self.limiter.ceiling = float(settings.limiter_ceiling)
        self.limiter.process(block)
        self.peak = float(np.max(np.abs(block)))
        return block

    def read_slot(self, slot: int) -> np.ndarray:
        """Read one mix block straight out of the GPU (offline path).

        Blocking, unlike the realtime readback: an offline render has no
        deadline, so waiting for the GPU is cheaper than managing fences.
        """
        out = np.zeros(AUDIO_BLOCK, dtype="f4")
        self.mix_buffer.read_into(out, size=AUDIO_BLOCK * 4,
                                  offset=slot * AUDIO_BLOCK * 4)
        return out

    def reset_offline(self, settings):
        """Reset conditioning state before an offline render."""
        self._ensure_gpu()
        self._envelope = 1.0
        # The offline path fills blocks across several physics steps, so it
        # uses the same slot cursor and partial-fill counter the realtime path
        # does. A leftover fill from a previous render would offset every
        # block of this one.
        self.w = self.r = 0
        self.block_fill = 0
        self.limiter = Limiter(int(settings.sample_rate),
                               ceiling=settings.limiter_ceiling)

    def update_telemetry(self, settings):
        settings.starves = self.starves
        settings.overruns = self.overruns
        settings.live_fences = self.live_fences
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
