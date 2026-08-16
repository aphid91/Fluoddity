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
from utilities.gl_helpers import read_shader, tryset, shader_prepend
from state.streamline_state import (
    MAX_STEPS, MAX_STREAMLINES, MAX_DISPATCHES_PER_FRAME,
    AUDIO_MAX_DISPATCHES_PER_FRAME,
)
from state.audio_state import AUDIO_BLOCK, AUDIO_SLOTS

# SSBO binding points. 0/2/3/4 are claimed by sim.py's entity and rule buffers.
PATH_BINDING = 5
STATE_BINDING = 6

# Invocations per workgroup; must match local_size_x in the compute shaders.
LOCAL_SIZE = 64

# Bytes per ParticleState record (std430): vec2 pos + vec2 vel + 2 uints
# + 4 floats of audio voice state.
STATE_STRIDE = 40

# glMemoryBarrier bit for SSBO writes becoming visible to later shaders.
# moderngl's ctx.memory_barrier() defaults to GL_ALL_BARRIER_BITS, which
# serialises the entire pipeline and flushes every cache. The tracer hot loop
# runs up to several hundred dispatches per frame in audio mode, so paying the
# full barrier each time costs far more than the dispatch itself; every
# dependency in that loop is SSBO -> SSBO.
GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000

# update() calls with no physics step before the canvas counts as frozen.
# One rendered frame issues several tracer updates inside a single physics
# step, so this has to tolerate a few; a paused simulation exceeds it
# immediately and keeps climbing.
STALE_UPDATES_BEFORE_FROZEN = 4


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
        self._field_phase = 0.0  # Position between physics steps, 0..1
        self._last_physics_frame = -1  # sim.frame_count at the last phase reset
        self._stale_updates = 0  # Consecutive updates with no physics step
        self._canvas_advancing = True  # False once the canvas looks frozen
        # Integration steps observed within the current physics interval, and
        # the smoothed measurement of how many an interval actually holds.
        self._steps_since_physics_frame = 0
        self._measured_spp = 0.0
        self._interp_valid = False  # Set per dispatch; see _set_trace_uniforms
        # Work owed to the tracer, in integration steps, when it is being
        # driven from inside the physics loop. See update_for_physics_step().
        self._step_debt = 0.0
        self._interleaved = False  # True while the physics loop drives us
        self.sim = None  # Set by the orchestrator; supplies the physics uniforms

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

    def _trace_source(self) -> str:
        """Assemble the tracer with the shared particle physics spliced in.

        Same idiom sim.py uses for entity_update: shader_prepend inserts after
        line 1, so the last prepend ends up outermost. Order is therefore
        physics, then fourier, then the defines - fourier must be declared
        before the physics that uses it, and both need ENTITY_COUNT and
        STREAMLINE_READONLY visible.
        """
        src = read_shader('shaders/streamline_trace.glsl')
        src = shader_prepend(src, read_shader('shaders/entity_physics.glsl'))
        src = shader_prepend(src, read_shader('shaders/fourier4_4.glsl'))
        # STREAMLINE_READONLY makes entity_physics skip the entity buffer
        # declaration and expose get_can_lerp(); ENTITY_COUNT is referenced by
        # the shared code even though streamers never index entities.
        src = shader_prepend(
            src, '#define STREAMLINE_READONLY 1\n#define ENTITY_COUNT 1\n')
        return src

    def _compile(self) -> bool:
        """Compile all three programs. Returns True if every one succeeded.

        On failure the previously working programs are left untouched, so a
        typo during live shader editing does not take the overlay down.
        """
        try:
            trace = self.ctx.compute_shader(self._trace_source())
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

    def _ssbo_barrier(self):
        """Order this dispatch's SSBO writes before the next dispatch's reads.

        Narrower than ctx.memory_barrier() (GL_ALL_BARRIER_BITS) - see the note
        on GL_SHADER_STORAGE_BARRIER_BIT. Every dependency in the tracer loop is
        SSBO -> SSBO: the ring and particle-state buffers.
        """
        self.ctx.memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT)

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

    @staticmethod
    def _field_split(samples_per_physics_step: float, block: int) -> int:
        """Steps per sub-dispatch so one dispatch fits in a physics interval.

        The two canvas textures only describe the latest physics interval, so
        interpolation is only meaningful while a dispatch stays inside it. A
        512-sample audio block spans several intervals at any useful physics
        rate (6.4 of them at 600 Hz), so the block is split into equal
        power-of-two chunks that each fit.

        Power-of-two keeps the block exactly divisible, so the audio ring
        still receives whole blocks and nothing downstream changes.
        """
        if samples_per_physics_step <= 0:
            return block
        steps = block
        while steps > 1 and steps > samples_per_physics_step:
            steps //= 2
        return max(1, steps)

    def _field_alpha(self, samples_per_physics_step: float, steps: int):
        """Blend fraction for this dispatch, and its per-step increment.

        The canvas advances one physics step at a time while the tracer runs
        many integration steps in between, so each step is placed
        proportionally through that interval.

        The phase carries across sub-dispatches. Restarting it at 0 each time
        made the field ramp toward the current frame and then snap back to the
        previous one at every sub-dispatch boundary - a sawtooth at the
        sub-dispatch rate. With a 512-sample block that is 750 Hz across the
        whole 400-600 Hz physics range, which is audible as a fixed tone that
        does not change with the config. Advancing the phase instead keeps the
        blend monotonic across the whole physics interval.

        The phase is reset in update() when a new physics step lands, since
        that is when the two canvas textures genuinely describe a new
        interval.
        """
        # Interpolation is only meaningful when the whole dispatch lands
        # inside the single physics interval the two canvas textures describe.
        self._interp_valid = max(samples_per_physics_step,
                                 self._measured_spp) >= steps
        if samples_per_physics_step <= 1.0:
            # The canvas moves at least as fast as the tracer; nothing to
            # interpolate.
            return 1.0, 0.0

        # Prefer the measured interval length; fall back to the caller's
        # estimate until one exists.
        spp = self._measured_spp if self._measured_spp > 1.0 else samples_per_physics_step
        self._steps_since_physics_frame += steps

        inc = 1.0 / spp
        base = self._field_phase
        end = base + inc * steps

        # The phase must not be cut short. samples_per_physics_step is an
        # estimate from smoothed render fps, so it disagrees with the real
        # cadence by a few percent and drifts run to run (measured 196..207
        # for a nominal 200). The frame_count watch in update() then reset a
        # ramp that was still mid-sweep - slamming the blend from ~0.95 back
        # to 0.0 - which is a discontinuity in the sensed field at exactly the
        # physics rate, and audible as a buzz there. Clamping at 1.0 instead
        # means an early reset lands on a blend that is already at the current
        # frame, so the reset is a no-op rather than a jump.
        self._field_phase = min(1.0, end)
        return base, inc

    def update_for_physics_step(self, canvas_texture, seed_world, settings,
                                steps_owed: float, audio_service=None,
                                audio_settings=None, prev_texture=None):
        """Advance the tracer inside a single physics step.

        This is the interleaved path, and it is what makes the field
        interpolation actually work at speedmult > 1.

        The canvas ping-pongs between two textures, so after a rendered frame
        has run all `speedmult` physics steps back to back, `canvas` and
        `canvas_prev` hold only frames N and N-1. A tracer that runs after the
        batch can therefore interpolate across exactly one interval and jumps
        blind over the other speedmult-1 - at speedmult=25 that is 24 of every
        25 field updates skipped, which is the "zippery above ~240 Hz" seam.
        The residual artifact sits at the RENDER rate, not the physics rate,
        which is why raising the physics rate never cured it.

        Called from SimulationRunner.post_physics_step_hook, the two textures
        genuinely describe the interval this call is inside, so the blend phase
        sweeps a true 0 -> 1 across it and every physics frame gets sensed.

        Args:
            steps_owed: Integration steps this physics step should produce.
                Fractional; the remainder is carried so the long-run rate is
                exact rather than quantised to whole steps per physics frame.

        Returns:
            Number of integration steps actually issued.
        """
        if self.trace_program is None or not settings.running:
            return 0

        audio_on = (audio_service is not None and audio_settings is not None
                    and audio_settings.enabled and audio_service.active)

        count = int(max(1, min(settings.count, MAX_STREAMLINES)))
        if count != self._last_count:
            self._needs_reset = True
        if settings.request_reset:
            settings.request_reset = False
            self._needs_reset = True
        if self._needs_reset:
            self._reset(seed_world, settings, count)

        self._step_debt += max(0.0, steps_owed)
        n_steps = int(self._step_debt)
        if n_steps <= 0:
            # Owed less than a whole step this interval - the debt carries, so
            # the long-run rate stays exact rather than being floored away.
            # Happens once the physics rate approaches the sample rate.
            return 0
        # Cap the catch-up after a hitch. One audio block is already several
        # physics intervals' worth of work, so anything beyond that is backlog
        # worth dropping rather than replaying against a stale field.
        if n_steps > AUDIO_BLOCK:
            n_steps = AUDIO_BLOCK
            self._step_debt = 0.0
        else:
            self._step_debt -= n_steps

        groups = (count + LOCAL_SIZE - 1) // LOCAL_SIZE

        # The whole point of this path: the two canvas textures describe THIS
        # interval, so interpolation is valid over all of it.
        self._interleaved = True
        self._interp_valid = True

        self._bind()
        self._set_trace_uniforms(settings, seed_world, n_steps, count,
                                 canvas_texture, prev_texture)
        tryset(self.trace_program, 'AUDIO_ENABLED', bool(audio_on))

        if not audio_on:
            # Visual only: one dispatch covering this interval's whole share.
            tryset(self.trace_program, 'STEPS_PER_DISPATCH', n_steps)
            # Only a decorrelation salt for the per-step hazard roll. Strided
            # by an odd constant rather than by the step count, so consecutive
            # dispatches cannot land on an index a previous one already used.
            tryset(self.trace_program, 'DISPATCH_INDEX',
                   (self._dispatch_count * 0x9E3779B9) & 0xFFFFFFFF)
            # Phase sweeps the full interval: step k of n_steps sits k/n_steps
            # of the way from the previous canvas frame to this one.
            tryset(self.trace_program, 'FIELD_ALPHA_BASE', 0.0)
            tryset(self.trace_program, 'FIELD_ALPHA_STEP', 1.0 / n_steps)
            self.trace_program.run(groups, 1, 1)
            self._ssbo_barrier()
            self._dispatch_count += 1
            self._interleaved = False
            return n_steps

        # Audio: the samples for this interval have to land contiguously in
        # the block being filled, so this walks the audio ring itself rather
        # than emitting one whole block per call. A physics interval is
        # usually a fraction of a block (at 600 Hz physics and 48 kHz audio it
        # is 80 samples of a 512-sample block), so a block spans several
        # intervals and is completed by whichever interval fills it.
        audio_service.bind()
        voice_count = int(min(max(audio_settings.voice_count, 1), count))
        tryset(self.trace_program, 'AUDIO_VOICE_COUNT', voice_count)
        tryset(self.trace_program, 'AUDIO_LANE_STRIDE', audio_service.lane_stride)
        tryset(self.trace_program, 'AUDIO_AMPLITUDE',
               1.0 if audio_settings.auto_gain else float(audio_settings.amplitude))
        tryset(self.trace_program, 'AUDIO_HP_COEFF',
               audio_service.highpass_coeff(audio_settings))
        tryset(self.trace_program, 'AUDIO_RAMP_DEC',
               audio_service.ramp_decrement(audio_settings))

        issued = 0
        remaining = n_steps
        while remaining > 0:
            if not audio_service.can_dispatch():
                # No free GPU slot; the readback has not caught up. Give the
                # unspent steps back so the rate stays honest.
                self._step_debt += remaining
                break
            # Fill only up to the end of the block in flight, so a dispatch
            # never straddles two audio slots.
            room = AUDIO_BLOCK - audio_service.block_fill
            chunk = min(remaining, room)

            tryset(self.trace_program, 'STEPS_PER_DISPATCH', chunk)
            tryset(self.trace_program, 'AUDIO_SLOT_BASE',
                   audio_service.slot_base + audio_service.block_fill)
            tryset(self.trace_program, 'DISPATCH_INDEX',
                   (self._dispatch_count * 0x9E3779B9 + issued) & 0xFFFFFFFF)
            # Where this chunk sits inside the physics interval. Consecutive
            # chunks of one interval continue the same ramp rather than each
            # restarting at 0 - restarting is what caused the 750 Hz sawtooth.
            base = float(n_steps - remaining) / n_steps
            tryset(self.trace_program, 'FIELD_ALPHA_BASE', base)
            tryset(self.trace_program, 'FIELD_ALPHA_STEP', 1.0 / n_steps)
            self.trace_program.run(groups, 1, 1)
            self._ssbo_barrier()

            remaining -= chunk
            issued += chunk
            if audio_service.advance_fill(chunk):
                # The block is now full: reduce it to mono and fence it, which
                # is what hands it to the readback.
                audio_service.reduce(audio_settings, voice_count)
                self._ssbo_barrier()
                audio_service.note_dispatch()

        self._dispatch_count += 1
        self._interleaved = False
        return issued

    def update(self, canvas_texture: moderngl.Texture,
               seed_world: tuple[float, float], settings, dt: float,
               audio_service=None, audio_settings=None,
               prev_texture=None, samples_per_physics_step: float = 0.0):
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

        # The two canvas textures describe the interval ending at the current
        # physics step, so the blend phase restarts only when that step
        # changes - not on every sub-dispatch, which would sawtooth.
        #
        # Tracking whether the canvas is advancing at all also matters: when
        # the simulation is paused the two textures freeze holding two
        # DIFFERENT frames (N and N-1), and a blend that keeps sweeping
        # between them makes the sensed field oscillate at the simulated
        # physics rate - audible as the audio being modulated at that rate
        # with the sim stopped. There is no interval to interpolate across
        # when nothing is moving.
        if self.sim is not None:
            fc = getattr(self.sim, 'frame_count', None)
            if fc is not None and fc != self._last_physics_frame:
                # Measure how many steps the last interval actually took,
                # rather than predicting it from render fps. The prediction
                # assumes a steady 60fps; when the app runs slower the physics
                # rate drops with it but the audio rate does not, so the
                # estimate came out several times too small and the ramp
                # finished long before the interval did.
                if self._last_physics_frame >= 0:
                    steps = self._steps_since_physics_frame
                    if steps > 0:
                        n = fc - self._last_physics_frame
                        measured = steps / max(1, n)
                        # Smoothed: the interval is not perfectly regular, and
                        # a jumpy estimate is itself a modulation.
                        self._measured_spp = (0.8 * self._measured_spp
                                              + 0.2 * measured
                                              if self._measured_spp > 0
                                              else measured)
                self._steps_since_physics_frame = 0
                self._last_physics_frame = fc
                self._field_phase = 0.0
                self._stale_updates = 0
            else:
                # Many dispatches legitimately land inside one physics step,
                # so a single stale update proves nothing. Only treat the
                # canvas as stopped once it has failed to advance for longer
                # than one interval could plausibly last.
                self._stale_updates += 1
        self._canvas_advancing = self._stale_updates <= STALE_UPDATES_BEFORE_FROZEN

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
            n = min(audio_service.blocks_wanted(dt),
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
        self._set_trace_uniforms(settings, seed_world, steps, count,
                                 canvas_texture, prev_texture)

        tryset(self.trace_program, 'AUDIO_ENABLED', bool(audio_on))
        if audio_on:
            audio_service.bind()
            voice_count = int(min(max(audio_settings.voice_count, 1), count))
            tryset(self.trace_program, 'AUDIO_VOICE_COUNT', voice_count)
            tryset(self.trace_program, 'AUDIO_LANE_STRIDE',
                   audio_service.lane_stride)
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
                # Split the block so each sub-dispatch stays inside one
                # physics interval; together they still fill the whole block.
                sub = self._field_split(samples_per_physics_step, steps)
                slot_base = audio_service.slot_base
                for off in range(0, steps, sub):
                    tryset(self.trace_program, 'STEPS_PER_DISPATCH', sub)
                    tryset(self.trace_program, 'AUDIO_SLOT_BASE', slot_base + off)
                    tryset(self.trace_program, 'DISPATCH_INDEX',
                           (self._dispatch_count * steps + off) & 0xFFFFFFFF)
                    base, inc = self._field_alpha(samples_per_physics_step, sub)
                    tryset(self.trace_program, 'FIELD_ALPHA_BASE', base)
                    tryset(self.trace_program, 'FIELD_ALPHA_STEP', inc)
                    self.trace_program.run(groups, 1, 1)
                    self._ssbo_barrier()
                # Restore for any later non-split use of this program.
                tryset(self.trace_program, 'STEPS_PER_DISPATCH', steps)
            else:
                # Advances every dispatch so the hazard roll never repeats,
                # even when the launch directions are deliberately frozen.
                tryset(self.trace_program, 'DISPATCH_INDEX',
                       (self._dispatch_count * steps) & 0xFFFFFFFF)
                base, inc = self._field_alpha(samples_per_physics_step, steps)
                tryset(self.trace_program, 'FIELD_ALPHA_BASE', base)
                tryset(self.trace_program, 'FIELD_ALPHA_STEP', inc)
                self.trace_program.run(groups, 1, 1)
            self._dispatch_count += 1
            issued += 1
            if audio_on:
                # The sub-dispatch loop above already barriered after its last
                # run, so the tracer's lane writes are visible to the reduction.
                audio_service.reduce(audio_settings, voice_count)
                # Order the reduction's writes before the readback, then fence
                # so the readback can poll without stalling the pipeline.
                self._ssbo_barrier()
                audio_service.note_dispatch()
            else:
                # Each dispatch reads the state the previous one wrote.
                self._ssbo_barrier()

        return issued

    def _set_trace_uniforms(self, settings, seed_world, steps, count,
                            canvas_texture, prev_texture=None):
        """Push the physics uniforms shared by the realtime and offline paths.

        Binds the canvas textures here rather than leaving it to the caller:
        the bind and the sampler uniform have to travel together, and
        separating them once already cost a bug where the realtime path
        sampled whatever happened to be in unit 0.
        """
        # Unit 1 is the canvas the shared physics reads (get_can); unit 2 is
        # the previous frame it interpolates from. Unit 5 is the field texture.
        canvas_texture.use(location=1)
        tryset(self.trace_program, 'canvas', 1)
        # Previous physics frame, for interpolating the field between steps.
        #
        # Only valid while a dispatch fits inside one physics interval: the
        # two textures describe just the latest interval, so a longer dispatch
        # would spend most of its samples clamped at the current frame with a
        # discontinuity at the clamp. Measured, that is worse than the
        # staircase it replaces, so it is gated rather than always on.
        interp = (settings.field_interpolation
                  and prev_texture is not None
                  and prev_texture is not canvas_texture
                  and self._interp_valid
                  # A frozen canvas has no interval to blend across; its two
                  # textures just hold two different old frames. Only the
                  # free-running path can observe that: the interleaved path
                  # is called BY the physics loop, so the canvas advanced by
                  # definition and the staleness counter never applies.
                  and (self._interleaved or self._canvas_advancing))
        (prev_texture if interp else canvas_texture).use(location=2)
        tryset(self.trace_program, 'canvas_prev', 2)
        tryset(self.trace_program, 'FIELD_INTERPOLATE', bool(interp))
        # The shared physics needs the same uniform values entity_update
        # gets. Sim owns that list, so it pushes them rather than this
        # service duplicating it.
        if self.sim is not None:
            self.sim.apply_physics_uniforms_to(self.trace_program,
                                               canvas_texture_unit=1)
        tryset(self.trace_program, 'seed_pos', tuple(seed_world))
        tryset(self.trace_program, 'STEPS_PER_DISPATCH', steps)
        tryset(self.trace_program, 'RING_CAPACITY', MAX_STEPS)
        tryset(self.trace_program, 'STREAMLINE_COUNT', count)
        tryset(self.trace_program, 'INITIAL_SPEED', float(settings.initial_speed))
        tryset(self.trace_program, 'STOP_AT_EDGE', bool(settings.stop_at_edge))
        tryset(self.trace_program, 'RESPAWN_AT_SEED', bool(settings.respawn_at_seed))
        tryset(self.trace_program, 'HAZARD_RATE',
               min(max(float(settings.hazard_rate), 0.0), 1.0))
        tryset(self.trace_program, 'SEED_SCATTER', float(settings.seed_scatter))
        # Newtonian mode and its four settings. Inert unless the mode is on.
        tryset(self.trace_program, 'NEWTONIAN_MODE',
               bool(settings.newtonian_mode))
        tryset(self.trace_program, 'FORCE_SCALE', float(settings.force_scale))
        tryset(self.trace_program, 'DAMPING', float(settings.damping))
        tryset(self.trace_program, 'STEP_SIZE', float(settings.step_size))
        tryset(self.trace_program, 'RESTORE_FORCE',
               float(settings.restore_force))
        tryset(self.trace_program, 'RUN_SALT', self._run_salt(settings))

    def render_audio_samples(self, canvas_texture, seed_world, settings,
                             audio_service, audio_settings, samples: int,
                             prev_texture=None):
        """Render `samples` audio samples for ONE physics step, while recording.

        The offline twin of update_for_physics_step(), and deliberately the
        same shape: called from inside the physics loop, it advances the tracer
        exactly `samples` integration steps against the interval the two canvas
        textures describe, so the blend phase sweeps a true 0 -> 1 across it.

        The previous version rendered whole 512-sample blocks. A physics step
        only owes 13-160 samples, so a step that finally owed a block rendered
        3 to 38 steps' worth of audio against one frozen canvas pair and the
        field then jumped forward all at once - the same staircase the realtime
        path had, which is why recordings still zippered after realtime was
        fixed. It also used _field_split()'s power-of-two rounding, which made
        a dispatch span a fractional number of physics intervals (0.6, 0.64,
        0.8) so phase and canvas drifted against each other.

        Returns a list of conditioned mono blocks, emitted only as each
        512-sample block completes; a call that does not finish a block
        returns [] and its samples stay in the block being filled.
        """
        if self.trace_program is None or samples <= 0:
            return []

        count = int(max(1, min(settings.count, MAX_STREAMLINES)))
        if count != self._last_count or self._needs_reset:
            self._reset(seed_world, settings, count)
        if settings.request_reset:
            settings.request_reset = False
            self._reset(seed_world, settings, count)

        voice_count = int(min(max(audio_settings.voice_count, 1), count))
        groups = (count + LOCAL_SIZE - 1) // LOCAL_SIZE

        # Interpolation is exact here for the same reason it is in realtime:
        # the caller is inside the physics step these textures describe.
        self._interleaved = True
        self._interp_valid = True

        self._bind()
        audio_service.bind()
        self._set_trace_uniforms(settings, seed_world, samples, count,
                                 canvas_texture, prev_texture)
        tryset(self.trace_program, 'AUDIO_ENABLED', True)
        tryset(self.trace_program, 'AUDIO_VOICE_COUNT', voice_count)
        tryset(self.trace_program, 'AUDIO_LANE_STRIDE', audio_service.lane_stride)
        tryset(self.trace_program, 'AUDIO_AMPLITUDE',
               1.0 if audio_settings.auto_gain else float(audio_settings.amplitude))
        tryset(self.trace_program, 'AUDIO_HP_COEFF',
               audio_service.highpass_coeff(audio_settings))
        tryset(self.trace_program, 'AUDIO_RAMP_DEC',
               audio_service.ramp_decrement(audio_settings))

        out = []
        remaining = samples
        issued = 0
        while remaining > 0:
            # Fill only to the end of the block in flight, so a dispatch never
            # straddles two slots. Same rule as the realtime path.
            room = AUDIO_BLOCK - audio_service.block_fill
            chunk = min(remaining, room)

            tryset(self.trace_program, 'STEPS_PER_DISPATCH', chunk)
            tryset(self.trace_program, 'AUDIO_SLOT_BASE',
                   audio_service.slot_base + audio_service.block_fill)
            tryset(self.trace_program, 'DISPATCH_INDEX',
                   (self._dispatch_count * 0x9E3779B9 + issued) & 0xFFFFFFFF)
            # Phase spans this physics interval, and consecutive chunks of the
            # same interval continue the ramp rather than restarting it.
            base = float(samples - remaining) / samples
            tryset(self.trace_program, 'FIELD_ALPHA_BASE', base)
            tryset(self.trace_program, 'FIELD_ALPHA_STEP', 1.0 / samples)
            self.trace_program.run(groups, 1, 1)
            self._ssbo_barrier()

            remaining -= chunk
            issued += chunk
            slot = audio_service.w % AUDIO_SLOTS
            if audio_service.advance_fill(chunk):
                # Block complete: reduce to mono, then read it back. Blocking
                # read - offline has no deadline, so this is simpler and
                # cheaper than fencing.
                audio_service.reduce(audio_settings, voice_count)
                self._ssbo_barrier()
                out.append(audio_service.condition_block(
                    audio_settings, audio_service.read_slot(slot)))
                audio_service.w += 1

        self._dispatch_count += 1
        self._interleaved = False
        return out

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
        # A real particle step is tiny compared with the canvas, so anything
        # approaching a quarter of the extent is a teleport (boundary wrap or
        # a hazard respawn), not motion.
        tryset(self.draw_program, 'JUMP_THRESHOLD', 0.5)
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
