"""Streamline Service - Traces and draws test particle paths through the field.

Test particles are released at the cursor and accelerated by the canvas vector
field. A compute shader integrates every streamline in parallel (one invocation
each, the whole path per invocation), writing positions into per-streamline
slices of an SSBO; an instanced line-strip pass then draws those slices.
"""
import moderngl
import numpy as np
from utilities.gl_helpers import read_shader, tryset
from state.streamline_state import MAX_STEPS, MAX_STREAMLINES

# SSBO binding point. 0/2/3/4 are claimed by sim.py's entity and rule buffers.
PATH_BINDING = 5

# Invocations per workgroup; must match local_size_x in streamline_trace.glsl.
LOCAL_SIZE = 64


class StreamlineService:
    """Traces streamlines from the cursor and renders them as an overlay."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.trace_program = None
        self.draw_program = None
        self.path_buffer = None
        self.vao = None
        self._frame = 0
        self._max_line_width = self._query_max_line_width()
        self._compile()
        self._allocate()

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
        """Compile both programs. Returns True if both succeeded.

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
            draw = self.ctx.program(
                vertex_shader=read_shader('shaders/streamline.vert'),
                fragment_shader=read_shader('shaders/streamline.frag'),
            )
        except Exception as e:
            print('Streamline draw shader compilation failed:')
            print(e)
            trace.release()
            return False

        # Both compiled - swap them in and retire the old ones.
        old_trace, old_draw, old_vao = self.trace_program, self.draw_program, self.vao
        self.trace_program = trace
        self.draw_program = draw
        # No vertex buffer: the vertex shader pulls positions from the SSBO
        # by gl_VertexID, so the VAO carries no attributes.
        self.vao = self.ctx.vertex_array(draw, [])

        if old_vao is not None:
            old_vao.release()
        if old_trace is not None:
            old_trace.release()
        if old_draw is not None:
            old_draw.release()
        return True

    def _allocate(self):
        """Allocate the path buffer: one slice of MAX_STEPS+1 vec2 per line."""
        if self.path_buffer is not None:
            return
        slots = MAX_STREAMLINES * (MAX_STEPS + 1)
        self.path_buffer = self.ctx.buffer(
            np.zeros((slots, 2), dtype=np.float32).tobytes()
        )

    def reload(self):
        """Recompile shaders. Called from command_handler on V key press."""
        if self._compile():
            print('Streamline shaders reloaded')

    def render(self, canvas_texture: moderngl.Texture, seed_world: tuple[float, float],
               cam_pos: tuple[float, float], cam_zoom: float,
               canvas_resolution: tuple[int, int], window_size: tuple[int, int],
               settings):
        """
        Trace and draw streamlines seeded at seed_world.

        Args:
            canvas_texture: Canvas texture holding the vector field (RG)
            seed_world: Seed position in world space [-1, 1]
            cam_pos: Camera position (x, y)
            cam_zoom: Camera zoom level
            canvas_resolution: Canvas texture resolution (width, height)
            window_size: Window size (width, height)
            settings: StreamlineState with integration and appearance params
        """
        if self.trace_program is None or self.draw_program is None:
            return

        steps = int(max(2, min(settings.steps, MAX_STEPS)))
        count = int(max(1, min(settings.count, MAX_STREAMLINES)))

        self._frame += 1
        # Holding the seed fixed freezes the spray's launch directions, so the
        # fan stays put and only the field's evolution moves it.
        random_seed = float(self._frame) * 0.618 if settings.resample_each_frame else 0.0

        # --- Pass 1: integrate every streamline into the SSBO ---
        canvas_texture.use(location=0)
        tryset(self.trace_program, 'canvas_texture', 0)
        tryset(self.trace_program, 'seed_pos', tuple(seed_world))
        tryset(self.trace_program, 'STEPS', steps)
        tryset(self.trace_program, 'MAX_STEPS', MAX_STEPS)
        tryset(self.trace_program, 'STREAMLINE_COUNT', count)
        tryset(self.trace_program, 'FORCE_SCALE', float(settings.force_scale))
        tryset(self.trace_program, 'DAMPING', float(settings.damping))
        tryset(self.trace_program, 'STEP_SIZE', float(settings.step_size))
        tryset(self.trace_program, 'RESTORE_FORCE', float(settings.restore_force))
        tryset(self.trace_program, 'INITIAL_SPEED', float(settings.initial_speed))
        tryset(self.trace_program, 'RANDOM_SEED', random_seed)

        self.path_buffer.bind_to_storage_buffer(PATH_BINDING)
        groups = (count + LOCAL_SIZE - 1) // LOCAL_SIZE
        self.trace_program.run(groups, 1, 1)

        # The draw pass reads what the compute pass just wrote.
        self.ctx.memory_barrier()

        # --- Pass 2: draw each slice as a line strip ---
        # Every instance is drawn with the same vertex count; the vertex shader
        # collapses and flags vertices past each streamline's own valid count,
        # so no readback (and no pipeline stall) is needed here.
        tryset(self.draw_program, 'cam_pos', cam_pos)
        tryset(self.draw_program, 'cam_zoom', cam_zoom)
        tryset(self.draw_program, 'canvas_resolution', canvas_resolution)
        tryset(self.draw_program, 'window_size', window_size)
        tryset(self.draw_program, 'MAX_STEPS', MAX_STEPS)
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

        self.vao.render(mode=moderngl.LINE_STRIP, vertices=steps, instances=count)

        if width != prev_width:
            self.ctx.line_width = prev_width

        self.ctx.disable(moderngl.BLEND)

    def cleanup(self):
        """Clean up GPU resources."""
        if self.vao is not None:
            self.vao.release()
        if self.trace_program is not None:
            self.trace_program.release()
        if self.draw_program is not None:
            self.draw_program.release()
        if self.path_buffer is not None:
            self.path_buffer.release()
