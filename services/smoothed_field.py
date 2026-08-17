"""Smoothed Field Service - a time-averaged copy of the canvas.

Maintains an exponential moving average of the canvas, updated once per
physics step:

    smoothed = mix(smoothed, canvas, amount)

and hands the tracer a (current, previous) pair with exactly the shape the
real canvas has, so the existing field interpolation works against it
unchanged.

Why: the canvas has strong structures that streamers fall into and resonate
with, but enough temporal noise that a tone only settles when the simulation
is paused and the tracer runs on a frozen field. Smoothing keeps the
structures alive long enough to ring while the field still evolves.

The pair is ping-ponged for the same reason the canvas is: the tracer
interpolates between the last two states, so the previous one has to survive
the write that produces the current one.
"""
import moderngl

from utilities.gl_helpers import read_shader, tryset


class SmoothedFieldService:
    """Owns the smoothed field's texture pair and its update pass."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = None
        self.vao = None
        self.textures = []      # [a, b], ping-ponged
        self.framebuffers = []
        # Index of the texture holding the newest smoothed state. The other
        # holds the previous one, which is what the tracer interpolates from.
        self.read_index = 0
        self._shape = None
        # Forces the next update to copy the canvas verbatim instead of
        # blending against whatever the textures happen to hold. Set on
        # allocate and on reset.
        self._needs_seed = True
        self._compile()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _compile(self) -> bool:
        """Compile the EMA pass. Leaves the old program alone on failure."""
        try:
            prog = self.ctx.program(
                vertex_shader=read_shader('shaders/canvas.vert'),
                fragment_shader=read_shader('shaders/smoothed_field.frag'),
            )
        except Exception as e:
            print('Smoothed field shader compilation failed:')
            print(e)
            return False

        old_prog, old_vao = self.program, self.vao
        self.program = prog
        # No vertex buffer: canvas.vert builds the fullscreen quad from
        # gl_VertexID, same as the canvas update pass.
        self.vao = self.ctx.vertex_array(prog, [])
        for obj in (old_vao, old_prog):
            if obj is not None:
                obj.release()
        return True

    def reload(self):
        """Recompile on the V-key shader reload."""
        if self._compile():
            print('Smoothed field shader reloaded')

    def ensure_allocated(self, shape: tuple[int, int]):
        """Allocate (or reallocate) the pair to match the canvas shape.

        The canvas is resized by world-size and aspect changes, so this tracks
        it rather than allocating once.
        """
        if self._shape == shape and self.textures:
            return
        self.release_textures()

        # Same format, filtering and wrapping as the canvas: the tracer
        # samples this through the identical code path, so any difference here
        # would change the field it senses for reasons unrelated to smoothing.
        self.textures = [
            self.ctx.texture(shape, 2, dtype='f4'),
            self.ctx.texture(shape, 2, dtype='f4'),
        ]
        for tex in self.textures:
            tex.repeat_x = True
            tex.repeat_y = True
        self.framebuffers = [
            self.ctx.framebuffer([self.textures[0]]),
            self.ctx.framebuffer([self.textures[1]]),
        ]
        for fb in self.framebuffers:
            fb.use()
            self.ctx.clear()
        self.read_index = 0
        self._shape = shape
        # Nothing meaningful in the textures yet; first update copies the
        # canvas rather than averaging against cleared memory.
        self._needs_seed = True

    def request_seed(self):
        """Reseed from the canvas on the next update.

        Used when continuity is meaningless - a simulation reset, or the
        smoothed field being switched on after running cold - so the tracer
        does not spend the first seconds on a field that is still catching up
        from a stale or empty state.
        """
        self._needs_seed = True

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    @property
    def current(self):
        """Newest smoothed state, or None before the first allocation."""
        if not self.textures:
            return None
        return self.textures[self.read_index]

    @property
    def previous(self):
        """Smoothed state as of the previous physics step."""
        if len(self.textures) < 2:
            return None
        return self.textures[1 - self.read_index]

    def is_ready(self) -> bool:
        """True once there is a pair the tracer can actually sample."""
        return self.program is not None and len(self.textures) == 2

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, canvas_texture, amount: float):
        """Advance the average by one physics step.

        Called from inside the physics loop, before the tracer runs, so the
        pair describes the same interval the tracer is stepping through -
        exactly the property the canvas pair has and the interpolation
        depends on.
        """
        if not self.is_ready() or canvas_texture is None:
            return

        # Write into the texture that is NOT current; it holds the oldest
        # state and is about to become the newest.
        write_index = 1 - self.read_index

        canvas_texture.use(location=1)
        tryset(self.program, 'real_canvas', 1)
        self.textures[self.read_index].use(location=2)
        tryset(self.program, 'prev_smoothed', 2)
        tryset(self.program, 'SMOOTH_AMOUNT', float(amount))
        tryset(self.program, 'SEED_FROM_CANVAS', bool(self._needs_seed))

        self.framebuffers[write_index].use()
        self.vao.render(moderngl.TRIANGLE_FAN, vertices=4)

        self.read_index = write_index
        self._needs_seed = False

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def release_textures(self):
        for fb in self.framebuffers:
            fb.release()
        for tex in self.textures:
            tex.release()
        self.framebuffers = []
        self.textures = []
        self._shape = None

    def cleanup(self):
        self.release_textures()
        for name in ('vao', 'program'):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
                setattr(self, name, None)
