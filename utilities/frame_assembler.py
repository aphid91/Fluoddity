import moderngl
import numpy as np
from utilities.gl_helpers import tryset, read_shader

def create_frame_assembly_shader(ctx, total_samples):
    """Create a shader program for frame assembly with temporal accumulation and gamma correction."""

    # Load shader source from files
    vertex_shader = read_shader('shaders/frame_assembly.vert')

    # Load fragment shader and replace total_samples placeholder
    fragment_shader = read_shader('shaders/frame_assembly.frag')
    fragment_shader = fragment_shader.replace('{total_samples}', str(total_samples))

    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def setup_frame_assembly(ctx, width, height, total_samples):
    """Set up GPU-based frame assembly system."""

    # Create accumulation texture and framebuffer
    accumulation_texture = ctx.texture((width, height), 4, dtype='f4')
    accumulation_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    accumulation_fbo = ctx.framebuffer(color_attachments=[accumulation_texture])

    # Create shader program
    shader = create_frame_assembly_shader(ctx, total_samples)

    # Create a fullscreen quad
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(shader, [(vbo, '2f', 'position')], ibo)

    return {
        'accumulation_fbo': accumulation_fbo,
        'accumulation_texture': accumulation_texture,
        'shader': shader,
        'vao': vao,
        'total_samples': total_samples,
        'width': width,
        'height': height
    }


class FrameAssembler:
    """GPU-based frame assembly with temporal accumulation and gamma correction."""

    def __init__(self, ctx, texture):
        """
        Initialize frame assembler.

        Args:
            ctx: moderngl.Context
            texture: The texture to assemble (determines size)
        """
        self.ctx = ctx
        self.width, self.height = texture.size
        self.resources = None

    def assemble_frame(self, input_texture, total_samples, current_sample_index, view_mode=0,
                       sweep_mode=False, sweep_reticle_pos=(0.5, 0.5), sweep_reticle_visible=False,
                       screen_aspect=1.0, brightness=1.0, exposure=0.0, ink_weight=1.0, watercolor_mode=False,
                       camera_position=(0.0, 0.0), camera_zoom=1.0,
                       trail_draw_radius=0.0,
                       mouse_screen_coords=(0.5, 0.5), tiling_mode=False, view_min=(0.0, 0.0),
                       view_max=(0.0, 0.0), tiling_scale=(1.0, 1.0), canvas_resolution=(1024, 1024),
                       tonemap_softness=1.0,
                       brush_mode=0, fixed_direction_heading=0.0,
                       field_texture=None, advanced_drawing_resources_initialized=False,
                       force_field_checked=False, strafe_field_checked=False,
                       draw_target_overlay_opacity=0.0):
        """
        Accumulate a frame and optionally apply gamma correction.

        Args:
            input_texture: moderngl.Texture to accumulate (PRE-gamma)
            total_samples: Number of frames in accumulation cycle
            current_sample_index: 0-indexed sample number (0 to total_samples-1)
            view_mode: Current view mode (0=can, 1=cam_brush)
            sweep_mode: Whether parameter sweeps are active
            sweep_reticle_pos: (x, y) screen UV position of sweep reticle
            sweep_reticle_visible: Whether to show the reticle
            screen_aspect: Screen width/height ratio for proper circle rendering
            brightness: Global brightness multiplier (applied before gamma)
            exposure: Frame blending amount (0=disabled, 1=long exposure)
            ink_weight: Watercolor mode optical density control
            watercolor_mode: Whether to use watercolor rendering
            camera_position: Camera position in world space (x, y)
            camera_zoom: Camera zoom level

        Returns:
            The assembled texture if final sample, None if still accumulating
        """
        # Check if we need to recreate resources
        input_width, input_height = input_texture.size
        recreate_resources = (
            self.resources is None or
            self.resources['total_samples'] != total_samples or
            self.resources['width'] != input_width or
            self.resources['height'] != input_height
        )

        if recreate_resources:
            # Clean up old resources if they exist
            if self.resources is not None:
                self.cleanup_resources()

            # Create new resources
            self.resources = setup_frame_assembly(
                self.ctx, input_width, input_height, total_samples
            )
            self.width = input_width
            self.height = input_height

        # Determine frame position in accumulation cycle
        is_first_frame = (current_sample_index == 0)
        final_sample = (current_sample_index == total_samples - 1)

        # Bind textures
        input_texture.use(location=0)  # input_frame
        self.resources['accumulation_texture'].use(location=1)  # accumulation_buffer
        if field_texture is not None:
            field_texture.use(location=3)  # field_texture

        # Set uniforms
        self.resources['shader']['input_frame'] = 0
        self.resources['shader']['accumulation_buffer'] = 1
        self.resources['shader']['is_first_frame'] = is_first_frame
        self.resources['shader']['final_sample'] = final_sample
        tryset(self.resources['shader'], 'view_mode', view_mode)
        tryset(self.resources['shader'], 'PARAMETER_SWEEP_MODE', sweep_mode)
        tryset(self.resources['shader'], 'sweep_reticle_pos', sweep_reticle_pos)
        tryset(self.resources['shader'], 'sweep_reticle_visible', sweep_reticle_visible)
        tryset(self.resources['shader'], 'screen_aspect', screen_aspect)
        tryset(self.resources['shader'], 'BRIGHTNESS', brightness)
        tryset(self.resources['shader'], 'EXPOSURE', exposure)
        tryset(self.resources['shader'], 'INK_WEIGHT', ink_weight)
        tryset(self.resources['shader'], 'WATERCOLOR_MODE', watercolor_mode)
        tryset(self.resources['shader'], 'TRAIL_DRAW_RADIUS', trail_draw_radius)
        tryset(self.resources['shader'], 'mouse_screen_coords', mouse_screen_coords)
        # Camera uniforms
        tryset(self.resources['shader'], 'camera_position', camera_position)
        tryset(self.resources['shader'], 'camera_zoom', camera_zoom)
        # Tiling mode uniforms
        tryset(self.resources['shader'], 'tiling_mode_enabled', tiling_mode)
        tryset(self.resources['shader'], 'view_min', view_min)
        tryset(self.resources['shader'], 'view_max', view_max)
        tryset(self.resources['shader'], 'tiling_scale', tiling_scale)
        tryset(self.resources['shader'], 'canvas_resolution', canvas_resolution)
        tryset(self.resources['shader'], 'TONEMAP_SOFTNESS', tonemap_softness)
        # Advanced drawing reticle uniforms
        tryset(self.resources['shader'], 'brush_mode', brush_mode)
        tryset(self.resources['shader'], 'fixed_direction_heading', fixed_direction_heading)
        # Advanced drawing field texture and checkbox state
        tryset(self.resources['shader'], 'field_texture', 3)
        tryset(self.resources['shader'], 'advanced_drawing_resources_initialized',
               advanced_drawing_resources_initialized)
        tryset(self.resources['shader'], 'force_field_checked', force_field_checked)
        tryset(self.resources['shader'], 'strafe_field_checked', strafe_field_checked)
        tryset(self.resources['shader'], 'draw_target_overlay_opacity', draw_target_overlay_opacity)

        # Render to accumulation buffer
        self.resources['accumulation_fbo'].use()
        #no need to clear the buffer, it's handled in frame_assembly.frag (is_first_frame)
        #if is_first_frame:
        #    self.resources['accumulation_fbo'].clear()
        self.resources['vao'].render()

        # Return assembled texture only on final sample
        if final_sample:
            return self.resources['accumulation_texture']

        return None

    def get_current_texture(self):
        """Get the current assembled texture (even if not fully assembled)."""
        if self.resources is None:
            return None
        return self.resources['accumulation_texture']

    def reset(self):
        """Clear the accumulation buffer."""
        if self.resources is not None:
            self.resources['accumulation_fbo'].clear()

    def cleanup_resources(self):
        """Clean up GPU resources."""
        if self.resources is not None:
            self.resources['accumulation_fbo'].release()
            self.resources['accumulation_texture'].release()
            self.resources['shader'].release()
            self.resources['vao'].release()
            self.resources = None

    def cleanup(self):
        """Clean up GPU resources when completely done."""
        self.cleanup_resources()
