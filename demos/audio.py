"""
GPU audio pipeline test-drive.

    compute shader -> SSBO slot ring -> fence -> read_into -> CPU limiter
        -> rtmixer lock-free ring -> PortAudio C callback

Single-threaded: audio blocks are produced in the render loop. The GPU ring
absorbs GPU-side jitter; the rtmixer ring absorbs render-loop jitter.

Deps:  pip install moderngl glfw imgui-bundle numpy rtmixer
Needs: OpenGL 4.3+ (compute shaders). Windows/Linux.
"""

import ctypes
import math
import sys
import time

import glfw
import moderngl
import numpy as np
import rtmixer
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

# ---------------------------------------------------------------------------
# Config.  Latency = (RB_BLOCKS + in-flight slots) * BLOCK / SR
# ---------------------------------------------------------------------------
SR = 48_000
BLOCK = 512          # frames per GPU dispatch  (10.7 ms)
CHANNELS = 2
LOCAL_SIZE = 64      # compute shader workgroup size; BLOCK must divide by it
SLOTS = 4            # GPU-side ring depth, in blocks
RB_BLOCKS = 8        # CPU-side ring depth, in blocks (must make a power of 2)

RB_FRAMES = BLOCK * RB_BLOCKS
assert RB_FRAMES & (RB_FRAMES - 1) == 0, "PortAudio ring size must be a power of 2"
assert BLOCK % LOCAL_SIZE == 0

TAU = 2.0 * math.pi
SLOT_BYTES = BLOCK * CHANNELS * 4

# ---------------------------------------------------------------------------
# GL sync objects.  moderngl 5.x wraps none of these, so pull them from the
# driver directly.  Note WINFUNCTYPE: GL uses __stdcall on Windows.
# ---------------------------------------------------------------------------
GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
GL_ALREADY_SIGNALED = 0x911A
GL_CONDITION_SATISFIED = 0x911C
GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001
GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000
GL_BUFFER_UPDATE_BARRIER_BIT = 0x00000200

_FUNC = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE


def _gl(name, restype, *argtypes):
    addr = glfw.get_proc_address(name)
    if not addr:
        raise RuntimeError(f"Driver did not export {name} (need GL 4.3+)")
    if not isinstance(addr, int):
        addr = ctypes.cast(addr, ctypes.c_void_p).value
    return _FUNC(restype, *argtypes)(addr)


class GLSync:
    """Loaded after the context exists."""

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
        """Non-blocking poll.  FLUSH_COMMANDS_BIT stops us waiting forever on
        commands still sitting unflushed in the driver queue."""
        r = self.client_wait(sync, GL_SYNC_FLUSH_COMMANDS_BIT, 0)
        return r in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED)

    def barrier(self):
        self.memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT
                            | GL_BUFFER_UPDATE_BARRIER_BIT)


# ---------------------------------------------------------------------------
# Compute shader.
#
# Key idea: the CPU owns the authoritative phase accumulator; the shader only
# computes the *within-block* phase delta.  Absolute sample index as a float
# would lose precision in minutes; a 512-sample tau never does.
#
# Vibrato is f(t) = freq + depth*sin(w_l*t + lfo0), whose phase integral is
# closed-form -- so every invocation is independent.  Real stateful DSP (filters,
# feedback) needs a prefix scan or serial-per-voice threads instead.
# ---------------------------------------------------------------------------
COMPUTE_SRC = f"""
#version 430
layout(local_size_x = {LOCAL_SIZE}) in;

layout(std430, binding = 0) writeonly buffer AudioOut {{
    vec2 samples[];
}};

uniform int   uSlot;
uniform float uPhase0;      // carrier phase at block start, wrapped to [0,TAU)
uniform float uLfoPhase0;
uniform float uFreq;
uniform float uDepth;       // vibrato depth, Hz
uniform float uLfoHz;
uniform float uAmp;

const float TAU = 6.28318530717958647692;
const float SR  = {float(SR)};
const int   BLK = {BLOCK};

void main() {{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(BLK)) return;

    float tau = float(i) / SR;
    float wl  = TAU * uLfoHz;

    // integral of freq + depth*sin(wl*t + lfo0) from 0 to tau
    float dphi = uFreq * tau
               + uDepth * (cos(uLfoPhase0) - cos(wl * tau + uLfoPhase0)) / wl;

    float s = uAmp * sin(uPhase0 + TAU * dphi);
    samples[uint(uSlot) * uint(BLK) + i] = vec2(s, s);
}}
"""


class Params:
    def __init__(self):
        self.freq = 220.0
        self.depth = 40.0     # vibrato depth in Hz
        self.lfo_hz = 5.0
        self.amp = 0.3
        self.drive = 1.0      # post-gain, to push the limiter


class GpuAudio:
    """SSBO ring of `SLOTS` blocks with a fence per slot."""

    def __init__(self, ctx, sync):
        self.ctx, self.sync = ctx, sync
        self.prog = ctx.compute_shader(COMPUTE_SRC)
        self.ssbo = ctx.buffer(reserve=SLOTS * SLOT_BYTES)
        self.ssbo.bind_to_storage_buffer(0)
        self.fences = [None] * SLOTS
        self.staging = np.zeros((BLOCK, CHANNELS), dtype="f4")
        self.w = 0            # next slot to dispatch
        self.r = 0            # next slot to read back
        self.phase = 0.0
        self.lfo_phase = 0.0

    def _set(self, name, value):
        u = self.prog.get(name, None)
        if u is not None:
            u.value = value

    def dispatch(self, p):
        slot = self.w % SLOTS
        self._set("uSlot", slot)
        self._set("uPhase0", self.phase)
        self._set("uLfoPhase0", self.lfo_phase)
        self._set("uFreq", p.freq)
        self._set("uDepth", p.depth)
        self._set("uLfoHz", max(p.lfo_hz, 0.01))
        self._set("uAmp", p.amp)

        self.prog.run(BLOCK // LOCAL_SIZE)
        self.sync.barrier()
        self.fences[slot] = self.sync.fence()

        # Advance the CPU-side accumulator by exactly what the shader did.
        # Phase stays continuous even when a slider moves mid-stream -> no click.
        tau_end = BLOCK / SR
        wl = TAU * max(p.lfo_hz, 0.01)
        dphi = (p.freq * tau_end
                + p.depth * (math.cos(self.lfo_phase)
                             - math.cos(wl * tau_end + self.lfo_phase)) / wl)
        self.phase = (self.phase + TAU * dphi) % TAU
        self.lfo_phase = (self.lfo_phase + wl * tau_end) % TAU
        self.w += 1

    def try_readback(self):
        """Return the staging array if the oldest in-flight block is ready."""
        if self.r >= self.w:
            return None
        slot = self.r % SLOTS
        if not self.sync.signalled(self.fences[slot]):
            return None
        # Fence already signalled, so this is a plain memcpy -- no stall.
        self.ssbo.read_into(self.staging, size=SLOT_BYTES, offset=slot * SLOT_BYTES)
        self.sync.delete_sync(self.fences[slot])
        self.fences[slot] = None
        self.r += 1
        return self.staging

    def in_flight(self):
        return self.w - self.r


class Limiter:
    """Block-rate peak limiter: instant attack, exponential release, gain
    linearly ramped across the block to avoid zipper noise.  A hard clip
    backstops it.  A real one needs lookahead -- see note in the README."""

    def __init__(self, ceiling=0.9, release_ms=120.0):
        self.ceiling = ceiling
        self.gain = 1.0
        self.release = math.exp(-1.0 / (release_ms * 0.001 * SR / BLOCK))
        self.reduction_db = 0.0

    def process(self, block):
        peak = float(np.max(np.abs(block))) or 1e-9
        desired = min(1.0, self.ceiling / peak)
        if desired < self.gain:
            target = desired                                    # attack: instant
        else:
            target = desired + (self.gain - desired) * self.release
        ramp = np.linspace(self.gain, target, BLOCK, dtype="f4")[:, None]
        block *= ramp
        np.clip(block, -1.0, 1.0, out=block)
        self.gain = target
        self.reduction_db = 20.0 * math.log10(max(target, 1e-6))


def pump(gpu, rb, params, limiter):
    """One iteration of the producer. Self-throttling: `r` only advances when
    the audio ring has room, which gates dispatch via the SLOTS invariant."""
    drained = 0
    while rb.write_available >= BLOCK:
        block = gpu.try_readback()
        if block is None:
            break
        limiter.process(block)
        block *= params.drive
        np.clip(block, -1.0, 1.0, out=block)
        rb.write(block)
        drained += 1
    while gpu.in_flight() < SLOTS - 1:
        gpu.dispatch(params)
    return drained


def main():
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(760, 560, "GPU audio pipeline", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context(require=430)
    sync = GLSync()
    gpu = GpuAudio(ctx, sync)
    params = Params()
    limiter = Limiter()

    imgui.create_context()
    impl = GlfwRenderer(window)

    rb = rtmixer.RingBuffer(4 * CHANNELS, RB_FRAMES)
    mixer = rtmixer.Mixer(channels=CHANNELS, samplerate=SR,
                          blocksize=256, latency="low")

    # Prefill before starting playback, otherwise PortAudio underruns instantly.
    while rb.write_available >= BLOCK:
        if pump(gpu, rb, params, limiter) == 0:
            time.sleep(0.001)

    mixer.start()
    mixer.play_ringbuffer(rb)

    vsync = True
    starves = 0
    frame_ms = 0.0
    stall_next = False

    while not glfw.window_should_close(window):
        t0 = time.perf_counter()
        glfw.poll_events()
        impl.process_inputs()

        if stall_next:            # simulate a frame hitch
            time.sleep(0.06)
            stall_next = False

        pump(gpu, rb, params, limiter)
        if rb.read_available < BLOCK:
            starves += 1

        imgui.new_frame()
        imgui.begin("Oscillator")
        _, params.freq = imgui.slider_float("Carrier Hz", params.freq, 40.0, 2000.0)
        _, params.depth = imgui.slider_float("Vibrato depth Hz", params.depth, 0.0, 400.0)
        _, params.lfo_hz = imgui.slider_float("Vibrato rate Hz", params.lfo_hz, 0.05, 20.0)
        _, params.amp = imgui.slider_float("Amp", params.amp, 0.0, 1.0)
        _, params.drive = imgui.slider_float("Drive (push limiter)", params.drive, 0.5, 8.0)
        imgui.end()

        imgui.begin("Pipeline")
        fill = rb.read_available / RB_FRAMES
        imgui.text(f"frame {frame_ms:5.2f} ms   in-flight {gpu.in_flight()}/{SLOTS}")
        imgui.text(f"latency ~{(RB_FRAMES + gpu.in_flight() * BLOCK) / SR * 1000:.0f} ms")
        imgui.progress_bar(fill, imgui.ImVec2(-1, 0), f"ring {fill * 100:.0f}%")
        imgui.text(f"starves: {starves}")
        imgui.text(f"limiter: {limiter.reduction_db:+.1f} dB")
        try:
            imgui.plot_lines("##wave", np.ascontiguousarray(gpu.staging[:, 0]),
                             scale_min=-1.0, scale_max=1.0,
                             graph_size=imgui.ImVec2(-1, 90))
        except Exception:
            pass
        changed, vsync = imgui.checkbox("vsync", vsync)
        if changed:
            glfw.swap_interval(1 if vsync else 0)
        if imgui.button("Stall 60 ms"):
            stall_next = True
        imgui.end()

        ctx.clear(0.07, 0.07, 0.09)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        frame_ms = (time.perf_counter() - t0) * 1000.0

    mixer.stop()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()