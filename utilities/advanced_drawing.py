"""Advanced Drawing processor: force/strafe field GPU texture management.

GPU resources are lazily initialized on first use, so there is zero
performance impact when advanced drawing is disabled.
"""

import numpy as np
import moderngl
from utilities.gl_helpers import read_shader, shader_prepend, tryset
from utilities.paths import get_user_data_dir, get_app_dir


class AdvancedDrawingProcessor:
    """Manages force/strafe field texture with lazy GPU resource creation.

    The field texture is RGBA float32 where:
      .xy = force field
      .zw = strafe field

    All GPU resources (shader, texture, FBO, VAO) are created lazily on the
    first call to ``process()`` and recreated if the canvas size changes.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._resources = None  # lazily created
        self._width = 0
        self._height = 0
        # Override shader resources (separate program/VAO for shader-driven field)
        self._override_resources = None
        self._override_shader_name = None  # currently compiled override shader filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def field_texture(self):
        """Return the force/strafe field texture, or None if not initialized."""
        if self._resources is None:
            return None
        return self._resources["field_tex"]

    def process(self, canvas_width, canvas_height,
                draw_mode, mouse_pos, prev_mouse_pos,
                draw_size, draw_power,
                brush_mode, fixed_direction_heading,
                force_field_active, strafe_field_active,
                tiling_mode, erase_mode,
                fill_mode=False, fill_direction_type=0):
        """Draw to the force/strafe field texture. Called once per render frame.

        Uses a two-pass approach:
          Pass 1 (erase): no blending, outputs vec4(0) in circle, discard outside
          Pass 2 (draw/fill): additive blending (ONE, ONE), outputs delta
        """
        self._ensure_resources(canvas_width, canvas_height)
        r = self._resources

        r["field_fbo"].use()

        # Set common uniforms
        tryset(r["program"], "mouse", mouse_pos)
        tryset(r["program"], "previous_mouse", prev_mouse_pos)
        tryset(r["program"], "draw_size", draw_size)
        tryset(r["program"], "draw_power", draw_power)
        tryset(r["program"], "brush_mode", brush_mode)
        tryset(r["program"], "fixed_direction_heading", fixed_direction_heading)
        tryset(r["program"], "force_field_active", force_field_active)
        tryset(r["program"], "strafe_field_active", strafe_field_active)
        tryset(r["program"], "tiling_mode", tiling_mode)

        # Pass 1: Erase (if right-click held) - no blending
        if erase_mode:
            tryset(r["program"], "erase_mode", True)
            tryset(r["program"], "draw_mode", False)
            tryset(r["program"], "fill_mode", False)
            self.ctx.disable(moderngl.BLEND)
            r["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        # Pass 2: Draw or fill (if active) - additive blending
        if draw_mode or fill_mode:
            tryset(r["program"], "erase_mode", False)
            tryset(r["program"], "draw_mode", draw_mode)
            tryset(r["program"], "fill_mode", fill_mode)
            tryset(r["program"], "fill_direction_type", fill_direction_type)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.ctx.blend_equation = moderngl.FUNC_ADD
            r["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)
            self.ctx.disable(moderngl.BLEND)

    def process_override(self, canvas_width, canvas_height, shader_name,
                         frame_count, mouse_pos, prev_mouse_pos,
                         draw_size, draw_power,
                         brush_mode, fixed_direction_heading,
                         tiling_mode,
                         camera_pos=(0.0, 0.0, 0.0),
                         camera_dir=(0.0, 0.0, 1.0),
                         defines_prefix=None):
        """Run the selected override shader to generate the field texture.

        Called once per render frame (not per physics step). Replaces the
        field texture contents entirely (no blending).
        """
        self._ensure_resources(canvas_width, canvas_height)
        self._ensure_override_resources(shader_name, defines_prefix=defines_prefix)
        r = self._resources
        ovr = self._override_resources
        if ovr is None:
            return

        r["field_fbo"].use()

        # Set uniforms (same as field_drawing.frag plus frame_count)
        tryset(ovr["program"], "canvas_resolution", (canvas_width, canvas_height))
        tryset(ovr["program"], "mouse", mouse_pos)
        tryset(ovr["program"], "previous_mouse", prev_mouse_pos)
        tryset(ovr["program"], "draw_size", draw_size)
        tryset(ovr["program"], "draw_power", draw_power)
        tryset(ovr["program"], "brush_mode", brush_mode)
        tryset(ovr["program"], "fixed_direction_heading", fixed_direction_heading)
        tryset(ovr["program"], "force_field_active", True)
        tryset(ovr["program"], "strafe_field_active", True)
        tryset(ovr["program"], "tiling_mode", tiling_mode)
        tryset(ovr["program"], "frame_count", frame_count)
        tryset(ovr["program"], "erase_mode", False)
        tryset(ovr["program"], "draw_mode", False)
        tryset(ovr["program"], "fill_mode", False)
        tryset(ovr["program"], "camera_pos", camera_pos)
        tryset(ovr["program"], "camera_dir", camera_dir)

        # Render with no blending (fully replace field contents)
        self.ctx.disable(moderngl.BLEND)
        ovr["vao"].render(mode=moderngl.TRIANGLE_FAN, vertices=4)

    def reload(self):
        """Recompile all shaders (field_drawing + any active override).

        Called from command_handler on V key press.
        """
        if self._resources is not None:
            # Recompile the main field_drawing program
            vert_src = read_shader("shaders/canvas.vert")
            frag_src = read_shader("shaders/field_drawing.frag")
            old_program = self._resources["program"]
            old_vao = self._resources["vao"]
            new_program = self.ctx.program(
                vertex_shader=vert_src, fragment_shader=frag_src)
            new_vao = self.ctx.vertex_array(new_program, [])
            tryset(new_program, "canvas_resolution", (self._width, self._height))
            self._resources["program"] = new_program
            self._resources["vao"] = new_vao
            old_vao.release()
            old_program.release()

        # Recompile override shader if one is active
        if self._override_resources is not None:
            shader_name = self._override_shader_name
            self._cleanup_override()
            if shader_name:
                self._ensure_override_resources(shader_name)

        print("field shaders reloaded")

    @staticmethod
    def get_available_override_shaders():
        """Return list of (filename, source) tuples for the override shader dropdown.

        Scans ~/Documents/Fluoddity/*.frag then shaders/field_override/*.frag.
        Returns a list of (display_name, is_divider) tuples where is_divider=True
        marks the separator between user and bundled shaders.
        """
        results = []

        # User shaders from Documents/Fluoddity
        user_dir = get_user_data_dir()
        if user_dir.exists():
            user_frags = sorted(user_dir.glob("*.frag"))
            for f in user_frags:
                results.append((f.name, False))

        # Divider (only if there are user shaders AND bundled shaders)
        bundled_dir = get_app_dir() / "shaders" / "field_override"
        bundled_frags = sorted(bundled_dir.glob("*.frag")) if bundled_dir.exists() else []

        if results and bundled_frags:
            results.append(("---", True))

        # Bundled shaders
        for f in bundled_frags:
            results.append((f.name, False))

        return results

    @staticmethod
    def resolve_override_shader_path(shader_name):
        """Resolve a shader filename to its full path.

        Checks user data dir first, then bundled shaders/field_override/.
        Returns the Path or None if not found.
        """
        user_path = get_user_data_dir() / shader_name
        if user_path.exists():
            return user_path
        bundled_path = get_app_dir() / "shaders" / "field_override" / shader_name
        if bundled_path.exists():
            return bundled_path
        return None

    def clear_fields(self):
        """Clear the force/strafe field texture to zero."""
        if self._resources is not None:
            self._resources["field_fbo"].clear()

    def clear_force_field(self):
        """Clear only the force field channels (.xy) to zero, preserving strafe (.zw)."""
        if self._resources is None:
            return
        tex = self._resources["field_tex"]
        data = np.frombuffer(tex.read(), dtype=np.float32).reshape(tex.height, tex.width, 4).copy()
        data[:, :, 0:2] = 0.0
        tex.write(data.tobytes())

    def clear_strafe_field(self):
        """Clear only the strafe field channels (.zw) to zero, preserving force (.xy)."""
        if self._resources is None:
            return
        tex = self._resources["field_tex"]
        data = np.frombuffer(tex.read(), dtype=np.float32).reshape(tex.height, tex.width, 4).copy()
        data[:, :, 2:4] = 0.0
        tex.write(data.tobytes())

    def snapshot_field_data(self) -> np.ndarray | None:
        """Readback field texture from GPU to CPU as float32 array.

        Returns:
            (height, width, 4) float32 numpy array, or None if not initialized.
        """
        if self._resources is None:
            return None
        tex = self._resources["field_tex"]
        return np.frombuffer(tex.read(), dtype=np.float32).reshape(tex.height, tex.width, 4).copy()

    def write_field_data(self, data: np.ndarray) -> None:
        """Write CPU float32 data to GPU field texture.

        Handles dimension mismatch via bilinear resize.
        No-op if GPU resources are not initialized.
        """
        if self._resources is None:
            return
        from utilities.field_texture_io import write_field_to_gpu
        write_field_to_gpu(self._resources["field_tex"], data)

    def ensure_initialized(self, canvas_dim_x: int,canvas_dim_y:int) -> None:
        """Lazily initialize GPU resources if not yet created.

        Called by the config system when loading a config that has field data
        but the advanced drawing system hasn't been used yet this session.
        """
        self._ensure_resources(canvas_dim_x, canvas_dim_y)

    def cleanup(self):
        """Release all GPU resources."""
        self._cleanup_override()
        if self._resources is None:
            return
        r = self._resources
        r["field_fbo"].release()
        r["field_tex"].release()
        r["program"].release()
        r["vao"].release()
        self._resources = None
        self._width = 0
        self._height = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_resources(self, w, h):
        if self._resources is not None and self._width == w and self._height == h:
            return
        self.cleanup()
        self._width = w
        self._height = h

        ctx = self.ctx

        # Reuse canvas.vert (gl_VertexID-based fullscreen quad, no VBO needed)
        vert_src = read_shader("shaders/canvas.vert")
        frag_src = read_shader("shaders/field_drawing.frag")

        program = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )

        # Empty VAO — canvas.vert generates vertices via gl_VertexID
        vao = ctx.vertex_array(program, [])

        # Force/Strafe field texture: RGBA f4
        field_tex = ctx.texture((w, h), 4, dtype="f4")
        field_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        field_tex.repeat_x = True
        field_tex.repeat_y = True
        field_fbo = ctx.framebuffer(color_attachments=[field_tex])
        field_fbo.clear()  # Start with zeros

        tryset(program, "canvas_resolution", (w, h))

        self._resources = dict(
            program=program, vao=vao,
            field_tex=field_tex, field_fbo=field_fbo,
        )

    def _ensure_override_resources(self, shader_name, defines_prefix=None):
        """Compile the override shader if not already compiled (or if shader changed)."""
        if (self._override_resources is not None
                and self._override_shader_name == shader_name):
            return
        self._cleanup_override()

        shader_path = self.resolve_override_shader_path(shader_name)
        if shader_path is None:
            print(f"Warning: override shader '{shader_name}' not found")
            return

        vert_src = read_shader("shaders/canvas.vert")
        frag_src = shader_path.read_text()
        if defines_prefix:
            frag_src = shader_prepend(frag_src, defines_prefix)

        try:
            program = self.ctx.program(
                vertex_shader=vert_src, fragment_shader=frag_src)
        except Exception as e:
            print(f"Error compiling override shader '{shader_name}': {e}")
            return

        vao = self.ctx.vertex_array(program, [])
        self._override_resources = dict(program=program, vao=vao)
        self._override_shader_name = shader_name

    def _cleanup_override(self):
        """Release override shader GPU resources."""
        if self._override_resources is None:
            return
        self._override_resources["vao"].release()
        self._override_resources["program"].release()
        self._override_resources = None
        self._override_shader_name = None
