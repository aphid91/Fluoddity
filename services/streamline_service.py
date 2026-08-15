"""Streamline Service - Stateful, ring-buffered particle tracing.

Particles persist between dispatches. Each dispatch advances every particle a
few integration steps and appends the new positions to that particle's ring
buffer; the draw pass renders the most recent slice of each ring as a line
strip.

The tracer runs on its own clock, independent of both the render loop and the
physics stepping. `update()` accumulates elapsed wall-clock time and issues as
many dispatches as that time is worth (capped per frame), which is the same
producer shape as demos/audio.py's pump(): a fixed, small amount of work per
dispatch, decoupled from frame cadence. That is what lets a future audio tap
emit one block per dispatch at a rate the audio device dictates.
"""
import moderngl
import numpy as np
from utilities.gl_helpers import read_shader, tryset
from state.streamline_state import (
    MAX_STEPS, MAX_STREAMLINES, MAX_DISPATCHES_PER_FRAME,
    AUDIO_MAX_DISPATCHES_PER_FRAME,
)

# SSBO binding points. 0/2/3/4 are claimed by sim.py's entity and rule buffers.
PATH_BINDING = 5
STATE_BINDING = 6

# Invocations per workgroup; must match local_size_x in the compute shaders.
LOCAL_SIZE = 64

# Bytes per ParticleState record (std430): vec2 pos + vec2 vel + 2 uints
# + 4 floats of audio voice state.
STATE_STRIDE = 40


class StreamlineService:
    """Traces persistent streamline particles and renders their recent paths."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.trace_program = None
        self.reset_program = None
        self.draw_program = None
        self.path_buffer = None
        self.state_buffer = None
        self.vao = None

        # Scheduling
        self._accumulator = 0.0  # Unspent time, in seconds
        self._dispatch_count = 0  # Total dispatches issued (drives reseeding)
        self._reset_count = 0  # Resets issued (salts the population RNG)
        self._last_count = 0  # Population size at the last reset
        self._needs_reset = True  # Force a seed before the first dispatch

        self._max_line_width = self._query_max_line_width()
        self._compile()
        self._allocate()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _query_max_line_width(self) -> float:
        """Largest line width the driver will actually rasterize.

        Core profile only guarantees 1.0, but NVIDIA honours the full aliased
        range. Reporting the real number lets the UI cap its slider instead of
        offering widths that silently do nothing.
        """
        try:
            rng = self.ctx.info.get('GL_ALIASED_LINE_WIDTH_RANGE')
            if rng:
                return float(max(1.0, rng[1]))
        except Exception:
            pass
        return 1.0

    @property
    def max_line_width(self) -> float:
        return self._max_line_width

    def _compile(self) -> bool:
        """Compile all three programs. Returns True if every one succeeded.

        On failure the previously working programs are left untouched, so a
        typo during live shader editing does not take the overlay down.
        """
        try:
            trace = self.ctx.compute_shader(read_shader('shaders/streamline_trace.glsl'))
        except Exception as e:
            print('Streamline trace shader compilation failed:')
            print(e)
            return False

        try:
            reset = self.ctx.compute_shader(read_shader('shaders/streamline_reset.glsl'))
        except Exception as e:
            print('Streamline reset shader compilation failed:')
            print(e)
            trace.release()
            return False

        try:
            draw = self.ctx.program(
                vertex_shader=read_shader('shaders/streamline.vert'),
                fragment_shader=read_shader('shaders/streamline.frag'),
            )
        except Exception as e:
            print('Streamline draw shader compilation failed:')
            print(e)
            trace.release()
            reset.release()
            return False

        old = (self.trace_program, self.reset_program, self.draw_program, self.vao)
        self.trace_program = trace
        self.reset_program = reset
        self.draw_program = draw
        # No vertex buffer: the vertex shader pulls positions from the ring by
        # gl_VertexID, so the VAO carries no attributes.
        self.vao = self.ctx.vertex_array(draw, [])

        for obj in old:
            if obj is not None:
                obj.release()
        return True

    def _allocate(self):
        """Allocate the ring and particle-state buffers."""
        if self.path_buffer is None:
            slots = MAX_STREAMLINES * MAX_STEPS
            self.path_buffer = self.ctx.buffer(
                np.zeros((slots, 2), dtype=np.float32).tobytes()
            )
        if self.state_buffer is None:
            self.state_buffer = self.ctx.buffer(
                np.zeros(MAX_STREAMLINES * STATE_STRIDE, dtype=np.uint8).tobytes()
            )

    def reload(self):
        """Recompile shaders. Called from command_handler on V key press."""
        if self._compile():
            print('Streamline shaders reloaded')

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def request_reset(self):
        """Re-seed every particle on the next update."""
        self._needs_reset = True

    def _bind(self):
        self.path_buffer.bind_to_storage_buffer(PATH_BINDING)
        self.state_buffer.bind_to_storage_buffer(STATE_BINDING)

    def _run_salt(self, settings) -> int:
        """Population-wide RNG salt.

        Advances on every reset so a fresh population gets fresh launch
        directions; held fixed when the user wants a repeatable fan.
        """
        if not settings.resample_each_frame:
            return 0
        return self._reset_count * 0x9E3779B9 & 0xFFFFFFFF

    def _reset(self, seed_world, settings, count):
        """Place every particle at its start point and clear its ring."""
        self._reset_count += 1
        self._bind()
        tryset(self.reset_program, 'seed_pos', tuple(seed_world))
        tryset(self.reset_program, 'RING_CAPACITY', MAX_STEPS)
        tryset(self.reset_program, 'STREAMLINE_COUNT', count)
        tryset(self.reset_program, 'INITIAL_SPEED', float(settings.initial_speed))
        tryset(self.reset_program, 'SEED_SCATTER', float(settings.seed_scatter))
        tryset(self.reset_program, 'RUN_SALT', self._run_salt(settings))
        self.reset_program.run((count + LOCAL_SIZE - 1) // LOCAL_SIZE, 1, 1)
        self.ctx.memory_barrier()
        self._needs_reset = False
        self._last_count = count

    def update(self, canvas_texture: moderngl.Texture,
               seed_world: tuple[float, float], settings, dt: float,
               audio_service=None, audio_settings=None):
        """Advance the tracer by however many dispatches `dt` is worth.

        Args:
            canvas_texture: Canvas texture holding the vector field (RG)
            seed_world: Seed position in world space [-1, 1]
            settings: StreamlineState
            dt: Wall-clock seconds since the last update
            audio_service: AudioService, when audio is driving the clock
            audio_settings: AudioState, when audio is driving the clock

        Returns:
            Number of dispatches issued this call.
        """
        if self.trace_program is None:
            return 0

        audio_on = (audio_service is not None and audio_settings is not None
                    and audio_settings.enabled and audio_service.active)

        count = int(max(1, min(settings.count, MAX_STREAMLINES)))

        # A population change invalidates the existing state records.
        if count != self._last_count:
            self._needs_reset = True

        if settings.request_reset:
            settings.request_reset = False
            self._needs_reset = True

        if self._needs_reset:
            self._reset(seed_world, settings, count)

        if not settings.running:
            self._accumulator = 0.0
            return 0

        if audio_on:
            # The sound device sets the pace, not the wall clock. Produce
            # exactly as many blocks as the audio ring has room for, bounded
            # by the free GPU slots. Under-producing is an audible gap, so
            # this deliberately does not "drop backlog" the way the visual
            # path does.
            self._accumulator = 0.0
            n = min(audio_service.blocks_wanted(),
                    AUDIO_MAX_DISPATCHES_PER_FRAME)
            if n <= 0:
                return 0
        else:
            # Convert elapsed time into a whole number of dispatches, keeping
            # the remainder so the long-run rate stays accurate.
            rate = max(1.0, float(settings.dispatch_hz))
            self._accumulator += max(0.0, dt)
            n = int(self._accumulator * rate)
            if n <= 0:
                return 0
            # Drop backlog beyond the cap rather than queueing a burst after a
            # hitch; catching up on stale time is worse than losing it.
            if n > MAX_DISPATCHES_PER_FRAME:
                n = MAX_DISPATCHES_PER_FRAME
                self._accumulator = 0.0
            else:
                self._accumulator -= n / rate

        steps = int(max(1, min(settings.steps_per_dispatch, MAX_STEPS)))
        groups = (count + LOCAL_SIZE - 1) // LOCAL_SIZE

        self._bind()
        canvas_texture.use(location=0)
        tryset(self.trace_program, 'canvas_texture', 0)
        tryset(self.trace_program, 'seed_pos', tuple(seed_world))
        tryset(self.trace_program, 'STEPS_PER_DISPATCH', steps)
        tryset(self.trace_program, 'RING_CAPACITY', MAX_STEPS)
        tryset(self.trace_program, 'STREAMLINE_COUNT', count)
        tryset(self.trace_program, 'FORCE_SCALE', float(settings.force_scale))
        tryset(self.trace_program, 'DAMPING', float(settings.damping))
        tryset(self.trace_program, 'STEP_SIZE', float(settings.step_size))
        tryset(self.trace_program, 'RESTORE_FORCE', float(settings.restore_force))
        tryset(self.trace_program, 'INITIAL_SPEED', float(settings.initial_speed))
        tryset(self.trace_program, 'STOP_AT_EDGE', bool(settings.stop_at_edge))
        tryset(self.trace_program, 'RESPAWN_AT_SEED', bool(settings.respawn_at_seed))
        tryset(self.trace_program, 'HAZARD_RATE',
               min(max(float(settings.hazard_rate), 0.0), 1.0))
        tryset(self.trace_program, 'SEED_SCATTER', float(settings.seed_scatter))
        tryset(self.trace_program, 'RUN_SALT', self._run_salt(settings))

        tryset(self.trace_program, 'AUDIO_ENABLED', bool(audio_on))
        if audio_on:
            audio_service.bind()
            tryset(self.trace_program, 'AUDIO_VOICE',
                   int(min(max(audio_settings.voice_index, 0), count - 1)))
            # With auto-gain the CPU applies amplitude after normalising, so
            # the shader must not scale as well or the gain would be squared.
            tryset(self.trace_program, 'AUDIO_AMPLITUDE',
                   1.0 if audio_settings.auto_gain
                   else float(audio_settings.amplitude))
            tryset(self.trace_program, 'AUDIO_HP_COEFF',
                   audio_service.highpass_coeff(audio_settings))
            tryset(self.trace_program, 'AUDIO_RAMP_DEC',
                   audio_service.ramp_decrement(audio_settings))

        issued = 0
        for _ in range(n):
            if audio_on and not audio_service.can_dispatch():
                # No free GPU slot: the readback has not caught up yet.
                break
            if audio_on:
                tryset(self.trace_program, 'AUDIO_SLOT_BASE',
                       audio_service.slot_base)
            # Advances every dispatch so the hazard roll never repeats, even
            # when the launch directions are deliberately frozen.
            tryset(self.trace_program, 'DISPATCH_INDEX',
                   (self._dispatch_count * steps) & 0xFFFFFFFF)
            self.trace_program.run(groups, 1, 1)
            self._dispatch_count += 1
            issued += 1
            if audio_on:
                # Fence this block so the readback can poll it without
                # stalling the pipeline.
                audio_service.note_dispatch()
            else:
                # Each dispatch reads the state the previous one wrote.
                self.ctx.memory_barrier()

        return issued

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(self, cam_pos: tuple[float, float], cam_zoom: float,
             canvas_resolution: tuple[int, int], window_size: tuple[int, int],
             settings):
        """Draw the recent tail of every particle's ring."""
        if self.draw_program is None:
            return

        count = int(max(1, min(settings.count, MAX_STREAMLINES)))
        tail = int(max(2, min(settings.tail_length, MAX_STEPS)))

        self._bind()
        tryset(self.draw_program, 'cam_pos', cam_pos)
        tryset(self.draw_program, 'cam_zoom', cam_zoom)
        tryset(self.draw_program, 'canvas_resolution', canvas_resolution)
        tryset(self.draw_program, 'window_size', window_size)
        tryset(self.draw_program, 'RING_CAPACITY', MAX_STEPS)
        tryset(self.draw_program, 'TAIL_LENGTH', tail)
        # A gap larger than this is a teleport (boundary wrap or hazard
        # respawn) rather than motion, and the strip is broken there. Scaled
        # off the integrator's own step so a large Step Size does not start
        # shredding legitimate segments; floored so a tiny step still leaves
        # room for the fastest particles.
        tryset(self.draw_program, 'JUMP_THRESHOLD',
               max(0.15, float(settings.step_size) * 25.0))
        tryset(self.draw_program, 'line_color', tuple(settings.color))
        tryset(self.draw_program, 'line_opacity', float(settings.opacity))

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # glLineWidth is global state, so restore it afterwards rather than
        # leaving a fat line width set for whatever draws next.
        width = min(max(1.0, float(settings.line_width)), self._max_line_width)
        prev_width = self.ctx.line_width
        if width != prev_width:
            self.ctx.line_width = width

        self.vao.render(mode=moderngl.LINE_STRIP, vertices=tail, instances=count)

        if width != prev_width:
            self.ctx.line_width = prev_width

        self.ctx.disable(moderngl.BLEND)

    def cleanup(self):
        """Clean up GPU resources."""
        for obj in (self.vao, self.trace_program, self.reset_program,
                    self.draw_program, self.path_buffer, self.state_buffer):
            if obj is not None:
                obj.release()
