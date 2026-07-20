import moderngl
import time
import math
import numpy as np
from utilities.gl_helpers import read_shader, shader_prepend, prepend_defines, tryset, set_rule_uniform
from state import SimState

# Global constants
SIZE_OF_ENTITY_STRUCT = 4*8  # 4 bytes per 32bit value. 8 values (pos:2, vel:2, hue:1, size:1, padding:2)
SIZE_OF_RULE_STRUCT = 4*4*20  # 4 bytes per float32. 4 floats per vec4. 20 vec4s per rule

class Sim:
    def __init__(self, ctx: moderngl.Context, world_size: float = 1.0, canvas_aspect_ratio: str = "1:1"):
        self.ctx = ctx
        self.world_size = world_size
        self.canvas_aspect_ratio = canvas_aspect_ratio
        self.entity_count = self.get_entity_count()
        self.time = 0.0
        self.start_time_stamp = time.time()
        self.frame_count = 0
        self.setup_simulation_state()
        self.setup_shaders()

        # View options (for UI combo box)
        self.view_options = [self.can_textures[self.can_read_index]]
        self.view_option_labels = ['DEBUG - Canvas (Persistent particle trails)']

        # Current state (will be updated by apply_state each frame)
        self._state = SimState()
        self._camera_state = None  # Will be set by apply_camera_state

        # Deferred rule buffer update mechanism (avoids 192MB/frame write cost)
        self._pending_rule_buffer_update = False  # Set true to trigger rule buffer write next frame
        self._pending_entity_id = None  # Entity ID to read back after rule buffer is written

        # Frame-constant entity uniforms only change when state does. apply_state()
        # (once per rendered frame) marks them dirty; entity_update() re-uploads them
        # on the next step and clears the flag, so the ~35 physics-setting tryset()
        # calls run once/frame instead of once/physics-step (speedmult x savings).
        self._entity_uniforms_dirty = True

    def get_entity_count(self) -> int:
        """Calculate entity count based on world size."""
        return int(600000 * self.world_size)

    def get_canvas_dimensions(self) -> tuple[int, int]:
        """Calculate canvas dimensions based on world size and aspect ratio."""
        # Parse aspect ratio string "W:H" into a scale factor
        # factor = sqrt(W/H) so that width*height = 1024^2 * world_size (area preserved)
        try:
            w, h = self.canvas_aspect_ratio.split(":")
            factor = math.sqrt(int(w) / int(h))
        except (ValueError, ZeroDivisionError):
            factor = 1.0
        return (int(1024 * factor * math.sqrt(self.world_size)), int(1024 / factor * math.sqrt(self.world_size)))

    @property
    def can(self):
        """The most recently completed canvas texture (read buffer)."""
        return self.can_textures[self.can_read_index]

    def setup_simulation_state(self):
        # Update entity_count in case world_size changed
        self.entity_count = self.get_entity_count()
        canvas_dim_x,canvas_dim_y = self.get_canvas_dimensions()
        canvas_shape = (canvas_dim_x, canvas_dim_y)

        # Allocate state buffers
        self.entities = self.ctx.buffer(reserve=self.entity_count * SIZE_OF_ENTITY_STRUCT)
        self.rule_buffer = self.ctx.buffer(reserve=self.entity_count * SIZE_OF_RULE_STRUCT)

        # Multi-load config buffer (64 configs * 248 bytes per config)
        # Each MultiLoadConfig struct: 9 PhysicsSetting (54 floats) + 6 ints + 2 floats = 248 bytes
        MULTI_LOAD_CONFIG_SIZE = 248
        MAX_MULTI_LOAD_CONFIGS = 64
        self.multi_load_buffer = self.ctx.buffer(reserve=MAX_MULTI_LOAD_CONFIGS * MULTI_LOAD_CONFIG_SIZE)

        # Multi-load rule buffer (64 rules * SIZE_OF_RULE_STRUCT bytes per rule)
        self.multi_load_rule_buffer = self.ctx.buffer(reserve=MAX_MULTI_LOAD_CONFIGS * SIZE_OF_RULE_STRUCT)

        # Bind entity and rule buffers
        self.entities.bind_to_storage_buffer(0)
        self.rule_buffer.bind_to_storage_buffer(2)
        self.multi_load_buffer.bind_to_storage_buffer(3)  # Binding 3 matches shader layout
        self.multi_load_rule_buffer.bind_to_storage_buffer(4)  # Binding 4 for multi-load rules

        # Create double-buffered canvas textures (2-channel RG32F: velocity only)
        # We ping-pong between these to avoid reading and writing the same texture
        self.can_textures = [
            self.ctx.texture(canvas_shape, 2, dtype='f4'),
            self.ctx.texture(canvas_shape, 2, dtype='f4')
        ]
        for tex in self.can_textures:
            tex.repeat_x = True
            tex.repeat_y = True
        self.can_framebuffers = [
            self.ctx.framebuffer([self.can_textures[0]]),
            self.ctx.framebuffer([self.can_textures[1]])
        ]
        self.can_read_index = 0  # Index of texture to read from (write to the other)

        # For camera to use (will be updated each frame to point to the most recently written buffer)
        self.view_tex = self.can_textures[self.can_read_index]

        # Clear both canvas buffers initially
        for fb in self.can_framebuffers:
            fb.use()
            self.ctx.clear()

        # View options for debug view modes
        self.view_options = [self.can_textures[self.can_read_index]]
    def setup_shaders(self):
        canvas_dim_x,canvas_dim_y = self.get_canvas_dimensions()
        canvas_shape = (canvas_dim_x, canvas_dim_y)

        # 1. Entity update compute shader
        self.entity_update_source = read_shader('shaders/entity_update.glsl')
        self.entity_update_source = shader_prepend(self.entity_update_source, read_shader('shaders/fourier4_4.glsl'))
        self.entity_update_source = prepend_defines(self.entity_update_source, self.entity_count)

        try:
            self.entity_update_program = self.ctx.compute_shader(self.entity_update_source)
        except Exception as e:
            print('Entity Update Compilation Failed:')
            print(e)

        tryset(self.entity_update_program, 'canvas_resolution', canvas_shape)
        tryset(self.entity_update_program, 'canvas', 1)
        tryset(self.entity_update_program, 'field_texture', 5)

        # 2. Brush update shaders (instanced rendering)
        self.brush_vertex_source = read_shader('shaders/brush.vert')
        self.brush_vertex_source = prepend_defines(self.brush_vertex_source, self.entity_count)
        self.brush_fragment_source = read_shader('shaders/brush.frag')

        try:
            self.brush_update_program = self.ctx.program(
                vertex_shader=self.brush_vertex_source,
                fragment_shader=self.brush_fragment_source
            )
        except Exception as e:
            print('Brush Update Compilation Failed:')
            print(e)

        self.brush_update_program['canvas_resolution'] = canvas_shape
        self.brush_vao = self.ctx.vertex_array(self.brush_update_program, [])

        # 3. Canvas update shaders (fullscreen quad)
        self.canvas_vertex_source = read_shader('shaders/canvas.vert')
        self.canvas_fragment_source = read_shader('shaders/canvas.frag')

        try:
            self.canvas_update_program = self.ctx.program(
                vertex_shader=self.canvas_vertex_source,
                fragment_shader=self.canvas_fragment_source
            )
        except Exception as e:
            print('Canvas Update Compilation Failed:')
            print(e)

        self.canvas_vao = self.ctx.vertex_array(self.canvas_update_program, [])
        tryset(self.canvas_update_program, 'canvas_resolution', canvas_shape)


    def entity_update(self, ctx: moderngl.Context, multi_load_service=None,
                      is_preview_active=False, field_texture_bound=False,
                      force_field_strength: float = 1.0,
                      strafe_field_strength: float = 1.0,
                      generics: tuple = None):
        '''
        Run a single physics update on all particles
        '''
        # --- Per-step uniforms (genuinely change every physics step) ---
        tryset(self.entity_update_program, 'frame_count', self.frame_count)
        # Only write rules to buffer when explicitly requested (avoids 192MB/frame cost)
        tryset(self.entity_update_program, 'WRITE_RULES', self._pending_rule_buffer_update)

        multi_load_active = (multi_load_service and multi_load_service.is_active()
                             and not is_preview_active)

        # Multi-load progress advances every step, so its uniforms refresh per step.
        # (The expensive SSBO write inside is already guarded by its own dirty flag.)
        if multi_load_active:
            self._set_multi_load_uniforms(multi_load_service)

        # --- Frame-constant uniforms: upload once per frame, not once per step ---
        # These derive from self._state / the frame's args, which don't change across
        # the speedmult physics steps in a rendered frame. apply_state() sets the flag.
        if self._entity_uniforms_dirty:
            tryset(self.entity_update_program, 'canvas', 1)
            tryset(self.entity_update_program, 'WORLD_SIZE', self.world_size)

            # Advanced drawing field texture
            tryset(self.entity_update_program, 'field_texture', 5)
            tryset(self.entity_update_program, 'advanced_drawing_resources_initialized', field_texture_bound)
            tryset(self.entity_update_program, 'force_field_strength', force_field_strength)
            tryset(self.entity_update_program, 'strafe_field_strength', strafe_field_strength)

            # Normal mode: set single config uniforms (multi-load handled above per-step)
            if not multi_load_active:
                tryset(self.entity_update_program, 'MULTILOAD_COUNT', 0)
                self._assign_physics_setting('AXIAL_FORCE_SETTING', self._state.AXIAL_FORCE, 'Axial Force', 'AXIAL_FORCE', -1.0, 1.0)
                self._assign_physics_setting('LATERAL_FORCE_SETTING', self._state.LATERAL_FORCE, 'Lateral Force', 'LATERAL_FORCE', -1.0, 1.0)
                self._assign_physics_setting('SENSOR_GAIN_SETTING', self._state.SENSOR_GAIN, 'Sensor Gain', 'SENSOR_GAIN', 0.0, 5.0)
                self._assign_physics_setting('MUTATION_SCALE_SETTING', self._state.MUTATION_SCALE, 'Mutation Scale', 'MUTATION_SCALE', -0.5, 0.5)
                self._assign_physics_setting('DRAG_SETTING', self._state.DRAG, 'Drag', 'DRAG', -1.0, 1.0)
                self._assign_physics_setting('STRAFE_POWER_SETTING', self._state.STRAFE_POWER, 'Strafe Power', 'STRAFE_POWER', 0.0, 0.5)
                self._assign_physics_setting('SENSOR_ANGLE_SETTING', self._state.SENSOR_ANGLE, 'Sensor Angle', 'SENSOR_ANGLE', -1.0, 1.0)
                self._assign_physics_setting('GLOBAL_FORCE_MULT_SETTING', self._state.GLOBAL_FORCE_MULT, 'Global Force Mult', 'GLOBAL_FORCE_MULT', 0.0, 2.0)
                self._assign_physics_setting('SENSOR_DISTANCE_SETTING', self._state.SENSOR_DISTANCE, 'Sensor Distance', 'SENSOR_DISTANCE', 0.0, 4.0)
                tryset(self.entity_update_program, 'DISABLE_SYMMETRY', self._state.DISABLE_SYMMETRY)
                tryset(self.entity_update_program, 'ABSOLUTE_ORIENTATION', self._state.ABSOLUTE_ORIENTATION)
                tryset(self.entity_update_program, 'ORIENTATION_MIX', self._state.ORIENTATION_MIX)
                # Rule seed from sim state (saved with physics configs)
                tryset(self.entity_update_program, 'RULE_SEED', self._state.rule_seed)

            #both modes: set global and conditionally global uniforms
            tryset(self.entity_update_program, 'BOUNDARY_CONDITIONS_MODE', self._state.boundary_conditions)
            tryset(self.entity_update_program, 'RESET_MODE', self._state.initial_conditions)
            tryset(self.entity_update_program, 'COHORTS', self._state.num_cohorts)
            self._assign_physics_setting('HAZARD_RATE_SETTING', self._state.HAZARD_RATE, 'Hazard Rate', 'HAZARD_RATE', 0.0, 0.05)

            # Appearance settings from sim state (now part of physics config)
            tryset(self.entity_update_program, 'HUE_SENSITIVITY', self._state.hue_sensitivity)
            tryset(self.entity_update_program, 'COLOR_BY_COHORT', self._state.color_by_cohort)

            # Generic scratch uniforms for live-coding
            if generics is not None:
                tryset(self.entity_update_program, 'generic03', generics[0:4])
                tryset(self.entity_update_program, 'generic47', generics[4:8])

            self._entity_uniforms_dirty = False

        num_workgroups = (self.entity_count + 63) // 64
        ctx.memory_barrier()
        self.entity_update_program.run(num_workgroups)

    def brush_update(self, ctx: moderngl.Context):
        """Render particles additively into the canvas write-target FBO.

        FBO must already be bound by update(). Brush pre-scales output by
        (1-p)/p so that after canvas decay (multiply by p), the net
        contribution is (1-p) * original_brush_value.
        """
        tryset(self.brush_update_program, 'frame_count', self.frame_count)
        tryset(self.brush_update_program, 'trail_persistence', self._state.TRAIL_PERSISTENCE)

        # Pure additive blending (no alpha channel in RG32F canvas)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ctx.blend_equation = moderngl.FUNC_ADD

        self.brush_vao.render(mode=moderngl.TRIANGLE_FAN, instances=self.entity_count, vertices=4)

    def can_update(self, draw_mode: bool = False, mouse_pos: tuple[float, float] = None,
                   prev_mouse_pos: tuple[float, float] = None, draw_size: float = 0.1, draw_power: float = 0.0,
                   multi_load_service=None, is_preview_active = False, tiling_mode: bool = False,
                   brush_mode: int = 0, fixed_direction_heading: float = 0.0,
                   erase_mode: bool = False, fill_mode: bool = False, fill_direction_type: int = 0,
                   canvas_draw_active: bool = True):
        """Apply canvas decay (old * trail_persistence) plus draw/erase/fill.

        FBO must already be bound by update(). Canvas read texture must
        already be bound at location 1.
        """
        # Boundary conditions mode for wrap behavior
        tryset(self.canvas_update_program, 'BOUNDARY_CONDITIONS_MODE', self._state.boundary_conditions)
        tryset(self.canvas_update_program, 'tiling_mode', tiling_mode)

        # Multi-load mode: calculate weighted average trail settings
        if multi_load_service and multi_load_service.is_active() and not is_preview_active:
            trail_persistence, trail_diffusion = self._calculate_weighted_trail_settings(multi_load_service)
        else:
            trail_persistence = self._state.TRAIL_PERSISTENCE
            trail_diffusion = self._state.TRAIL_DIFFUSION

        # Assign TRAIL_PERSISTENCE as a PhysicsSetting struct
        min_val, max_val = self._get_slider_range('Trail Persistence', 0.0, 1.0)
        tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.slider_value', trail_persistence)
        tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.min_value', min_val)
        tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.max_value', max_val)
        # Only apply sweeps if parameter sweeps UI is enabled AND not in multi-load mode
        if self._state.parameter_sweeps_enabled and not (multi_load_service and multi_load_service.is_active()):
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.x_sweep', self._state.x_sweeps.get('TRAIL_PERSISTENCE', 0.0))
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.y_sweep', self._state.y_sweeps.get('TRAIL_PERSISTENCE', 0.0))
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.cohort_sweep', self._state.cohort_sweeps.get('TRAIL_PERSISTENCE', 0.0))
        else:
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.x_sweep', 0.0)
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.y_sweep', 0.0)
            tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.cohort_sweep', 0.0)
        # Always apply jitter (independent of parameter_sweeps_enabled)
        tryset(self.canvas_update_program, 'TRAIL_PERSISTENCE_SETTING.jitter', self._state.jitters.get('TRAIL_PERSISTENCE', 0.0))

        # Assign TRAIL_DIFFUSION as a PhysicsSetting struct
        min_val, max_val = self._get_slider_range('Trail Diffusion', 0.0, 1.0)
        tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.slider_value', trail_diffusion)
        tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.min_value', min_val)
        tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.max_value', max_val)
        # Only apply sweeps if parameter sweeps UI is enabled AND not in multi-load mode
        if self._state.parameter_sweeps_enabled and not (multi_load_service and multi_load_service.is_active()):
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.x_sweep', self._state.x_sweeps.get('TRAIL_DIFFUSION', 0.0))
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.y_sweep', self._state.y_sweeps.get('TRAIL_DIFFUSION', 0.0))
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.cohort_sweep', self._state.cohort_sweeps.get('TRAIL_DIFFUSION', 0.0))
        else:
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.x_sweep', 0.0)
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.y_sweep', 0.0)
            tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.cohort_sweep', 0.0)
        # Always apply jitter (independent of parameter_sweeps_enabled)
        tryset(self.canvas_update_program, 'TRAIL_DIFFUSION_SETTING.jitter', self._state.jitters.get('TRAIL_DIFFUSION', 0.0))

        tryset(self.canvas_update_program, 'can_tex', 1)

        # Pass frame count to shader for initialization
        tryset(self.canvas_update_program, 'frame_count', self.frame_count)

        # Set draw mode uniforms if in draw mode
        tryset(self.canvas_update_program, 'draw_mode', draw_mode)
        tryset(self.canvas_update_program, 'brush_mode', brush_mode)
        tryset(self.canvas_update_program, 'fixed_direction_heading', fixed_direction_heading)
        tryset(self.canvas_update_program, 'erase_mode', erase_mode)
        tryset(self.canvas_update_program, 'fill_mode', fill_mode)
        tryset(self.canvas_update_program, 'fill_direction_type', fill_direction_type)
        tryset(self.canvas_update_program, 'canvas_draw_active', canvas_draw_active)
        if (draw_mode or erase_mode or fill_mode) and mouse_pos is not None and prev_mouse_pos is not None:
            tryset(self.canvas_update_program, 'mouse', mouse_pos)
            tryset(self.canvas_update_program, 'previous_mouse', prev_mouse_pos)
            tryset(self.canvas_update_program, 'draw_size', draw_size)
            tryset(self.canvas_update_program, 'draw_power', draw_power)

        # FBO already bound by update() — just render the fullscreen quad
        self.canvas_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

    def update(self, ctx, draw_mode: bool = False, mouse_pos: tuple[float, float] = None,
               prev_mouse_pos: tuple[float, float] = None, draw_size: float = 0.1, draw_power: float = 0.0,
               multi_load_service=None, is_preview_active = False, tiling_mode: bool = False,
               strong_determinism: bool = False,
               brush_mode: int = 0, fixed_direction_heading: float = 0.0,
               erase_mode: bool = False, fill_mode: bool = False, fill_direction_type: int = 0,
               canvas_draw_active: bool = True,
               field_texture=None,
               force_field_strength: float = 1.0,
               strafe_field_strength: float = 1.0,
               generics: tuple = None):
        # Bind the current read buffer for entity_update sampling
        self.can_textures[self.can_read_index].use(location=1)

        # Bind advanced drawing field texture if available
        if field_texture is not None:
            field_texture.use(location=5)

        current_time = time.time()
        self.time = current_time - self.start_time_stamp

        # 1. Entity physics (reads canvas read buffer for sensors)
        self.entity_update(ctx, multi_load_service, is_preview_active,
                           field_texture_bound=field_texture is not None,
                           force_field_strength=force_field_strength,
                           strafe_field_strength=strafe_field_strength,
                           generics=generics)

        # 2. Determine write target for canvas decay + brush
        if strong_determinism:
            write_index = 1 - self.can_read_index
        else:
            write_index = self.can_read_index

        # 3. Canvas decay: read from read_index, write decayed values to write_index
        self.can_framebuffers[write_index].use()
        ctx.disable(moderngl.BLEND)
        self.can_update(draw_mode, mouse_pos, prev_mouse_pos, draw_size, draw_power,
                        multi_load_service, is_preview_active, tiling_mode,
                        brush_mode, fixed_direction_heading, erase_mode, fill_mode,
                        fill_direction_type, canvas_draw_active)

        # 4. Brush: additive render into same write_index FBO
        ctx.memory_barrier()
        self.brush_update(ctx)
        ctx.disable(moderngl.BLEND)

        # 5. Swap buffers if double-buffering
        if strong_determinism:
            self.can_read_index = write_index
            self.view_options[0] = self.can_textures[self.can_read_index]
            if self._state.current_view_option == 0:
                self.view_tex = self.can_textures[self.can_read_index]

        self.frame_count += 1

        # Increment multi-load progress if active
        if multi_load_service and multi_load_service.is_active():
            multi_load_service.increment_progress()

    def clear_canvas(self):
        """Clear only the trail/canvas textures (not particles or frame count)."""
        old_fbo = self.ctx.fbo
        for fb in self.can_framebuffers:
            fb.use()
            self.ctx.clear(0, 0, 0, 0)
        old_fbo.use()

    def reset(self):
        old_fbo = self.ctx.fbo
        # Clear both canvas buffers
        for fb in self.can_framebuffers:
            fb.use()
            self.ctx.clear(0, 0, 0, 0)
        self.frame_count = 0
        old_fbo.use()

    def reload(self):
        print('reloading shaders')
        self.setup_shaders()
        print('reload done')

    def apply_state(self, state: SimState) -> None:
        """Apply state from Orchestrator before update."""
        self._state = state
        # New state -> frame-constant entity uniforms need re-uploading once this frame.
        self._entity_uniforms_dirty = True
        # Update view_tex based on current_view_option
        if state.current_view_option < len(self.view_options):
            self.view_tex = self.view_options[state.current_view_option]

    def apply_camera_state(self, camera_state) -> None:
        """Apply camera state from Orchestrator before update."""
        self._camera_state = camera_state

    def _get_slider_range(self, slider_label: str, default_min: float, default_max: float) -> tuple[float, float]:
        """Get the current min/max range for a slider from sim state."""
        if self._state is None:
            return (default_min, default_max)

        if slider_label not in self._state.slider_ranges:
            return (default_min, default_max)

        return (self._state.slider_ranges[slider_label][0],
                self._state.slider_ranges[slider_label][1])

    def calculate_setting(self, slider_value: float, min_value: float, max_value: float,
                         pos: tuple[float, float], cohort: float,
                         x_sweep: float, y_sweep: float, cohort_sweep: float) -> float:
        """Python version of GLSL calculate_setting() function.

        SYNCHRONIZED: This function must match entity_update.glsl and canvas.frag
        Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py

        Calculates the effective parameter value based on sweeps and position/cohort.
        Mirrors the shader function for use when clicking particles to set slider values.

        Args:
            slider_value: Base slider value when no sweeps are active
            min_value: Minimum value for parameter sweeps
            max_value: Maximum value for parameter sweeps
            pos: (x, y) world position of entity in [-1, 1] range
            cohort: Raw cohort value in [0, num_cohorts) range (from entity buffer)
            x_sweep: Sweep mode (0.0 = off, 1.0 = normal, -1.0 = inverse)
            y_sweep: Sweep mode (0.0 = off, 1.0 = normal, -1.0 = inverse)
            cohort_sweep: Sweep mode (0.0 = off, 1.0 = normal, -1.0 = inverse)

        Returns:
            Effective parameter value at the given position/cohort
        """
        import math

        # If no sweeps active, return slider value
        if x_sweep == 0.0 and y_sweep == 0.0 and cohort_sweep == 0.0:
            return slider_value

        # Convert pos from [-1, 1] to [0, 1] for mixing
        pos_norm = ((pos[0] + 1) / 2, (pos[1] + 1) / 2)

        # Convert cohort to normalized [0, 1] range, matching shader:
        # cohort = floor(cohort) / float(get_particle_cohorts())
        cohort_norm = math.floor(cohort*self._state.num_cohorts) / float(self._state.num_cohorts)

        # Accumulate sweep contributions
        result = 0.0
        active_sweeps = 0

        if x_sweep != 0.0:
            # For inverse sweep (x_sweep < 0), swap min and max
            if x_sweep > 0.0:
                result += min_value + (max_value - min_value) * pos_norm[0]
            else:
                result += max_value + (min_value - max_value) * pos_norm[0]
            active_sweeps += 1

        if y_sweep != 0.0:
            # For inverse sweep (y_sweep < 0), swap min and max
            if y_sweep > 0.0:
                result += min_value + (max_value - min_value) * pos_norm[1]
            else:
                result += max_value + (min_value - max_value) * pos_norm[1]
            active_sweeps += 1

        if cohort_sweep != 0.0:
            # For inverse sweep (cohort_sweep < 0), swap min and max
            if cohort_sweep > 0.0:
                result += min_value + (max_value - min_value) * cohort_norm
            else:
                result += max_value + (min_value - max_value) * cohort_norm
            active_sweeps += 1

        # Average the results to keep within min/max range
        return result / active_sweeps if active_sweeps > 0 else slider_value

    def _assign_physics_setting(self, uniform_name: str, slider_value: float, slider_label: str, param_name: str, default_min: float, default_max: float):
        """Assign a PhysicsSetting struct uniform with dynamically fetched min/max ranges, sweep states, and jitter."""
        min_val, max_val = self._get_slider_range(slider_label, default_min, default_max)

        tryset(self.entity_update_program, f'{uniform_name}.slider_value', slider_value)
        tryset(self.entity_update_program, f'{uniform_name}.min_value', min_val)
        tryset(self.entity_update_program, f'{uniform_name}.max_value', max_val)
        # Only apply sweeps if parameter sweeps UI is enabled
        if self._state.parameter_sweeps_enabled:
            tryset(self.entity_update_program, f'{uniform_name}.x_sweep', self._state.x_sweeps.get(param_name, 0.0))
            tryset(self.entity_update_program, f'{uniform_name}.y_sweep', self._state.y_sweeps.get(param_name, 0.0))
            tryset(self.entity_update_program, f'{uniform_name}.cohort_sweep', self._state.cohort_sweeps.get(param_name, 0.0))
        else:
            tryset(self.entity_update_program, f'{uniform_name}.x_sweep', 0.0)
            tryset(self.entity_update_program, f'{uniform_name}.y_sweep', 0.0)
            tryset(self.entity_update_program, f'{uniform_name}.cohort_sweep', 0.0)
        # Always apply jitter (independent of parameter_sweeps_enabled)
        tryset(self.entity_update_program, f'{uniform_name}.jitter', self._state.jitters.get(param_name, 0.0))

    def _calculate_weighted_trail_settings(self, multi_load_service) -> tuple[float, float]:
        """Calculate weighted average trail settings based on multi-load window.

        The window is defined by current_progress (position in circular buffer, 0-1)
        and simultaneous_configs (span width in number of configs). We calculate
        which configs the window touches and their weights, then return weighted averages.

        Args:
            multi_load_service: MultiLoadService instance

        Returns:
            (trail_persistence, trail_diffusion) tuple of weighted averages
        """
        config_count = multi_load_service.get_config_count()
        if config_count == 0:
            return (0.938, 1.0)  # Default values

        current_progress = multi_load_service.current_progress
        simultaneous = multi_load_service.simultaneous_configs

        # Calculate window center and half-width in config index space
        # Each config occupies unit width [i, i+1) in index space
        half_width = simultaneous / 2.0 + 1e-3
        center = current_progress * config_count + half_width
        
        # Calculate weighted sum
        total_weight = 0.0
        weighted_persistence = 0.0
        weighted_diffusion = 0.0

        for i in range(config_count):
            # Calculate overlap between window and config i
            # Config i occupies space [i, i+1) in index space
            overlap = self._calculate_circular_overlap(
                center - half_width,  # window start
                center + half_width,  # window end
                float(i),              # config start
                float(i + 1),          # config end
                float(config_count)    # total configs for wrapping
            )

            if overlap > 0:
                config = multi_load_service.get_config(i)
                if config:
                    weighted_persistence += overlap * config.trail_persistence
                    weighted_diffusion += overlap * config.trail_diffusion
                    total_weight += overlap

        # Return weighted averages
        if total_weight > 0:
            return (weighted_persistence / total_weight, weighted_diffusion / total_weight)
        else:
            # Fallback to first config if no overlap (shouldn't happen)
            config = multi_load_service.get_config(0)
            if config:
                return (config.trail_persistence, config.trail_diffusion)
            return (0.938, 1.0)

    def _calculate_circular_overlap(self, win_start: float, win_end: float,
                                     cfg_start: float, cfg_end: float,
                                     total_count: float) -> float:
        """Calculate overlap between window and config in circular buffer.

        Args:
            win_start, win_end: Window bounds in index space (can be negative or > total_count)
            cfg_start, cfg_end: Config bounds in index space [i, i+1)
            total_count: Total number of configs

        Returns:
            Overlap amount (0 to 1.0 representing fraction of window)
        """
        # Normalize window bounds to [0, total_count) range with wrapping
        win_start = win_start % total_count
        win_end = win_end % total_count

        overlap = 0.0

        # Case 1: Window doesn't wrap (win_start < win_end)
        if win_start <= win_end:
            # Simple overlap calculation
            overlap_start = max(win_start, cfg_start)
            overlap_end = min(win_end, cfg_end)
            overlap = max(0.0, overlap_end - overlap_start)
        else:
            # Case 2: Window wraps around (win_start > win_end in normalized space)
            # The window consists of two segments: [win_start, total_count) and [0, win_end)

            # Check overlap with first segment [win_start, total_count)
            if cfg_end > win_start:
                overlap_start = max(win_start, cfg_start)
                overlap_end = min(total_count, cfg_end)
                overlap += max(0.0, overlap_end - overlap_start)

            # Check overlap with second segment [0, win_end)
            if cfg_start < win_end:
                overlap_start = max(0.0, cfg_start)
                overlap_end = min(win_end, cfg_end)
                overlap += max(0.0, overlap_end - overlap_start)

        return overlap

    def _set_multiload_physics_param(self, array_name: str, index: int, config, param_attr: str, slider_label: str, param_name: str, default_min: float, default_max: float):
        """Helper to set a single PhysicsSetting struct in an array for multi-load mode."""
        slider_value = getattr(config, param_attr)
        min_val, max_val = self._get_slider_range(slider_label, default_min, default_max)

        tryset(self.entity_update_program, f'{array_name}[{index}].slider_value', slider_value)
        tryset(self.entity_update_program, f'{array_name}[{index}].min_value', min_val)
        tryset(self.entity_update_program, f'{array_name}[{index}].max_value', max_val)
        # Include sweep data from config
        tryset(self.entity_update_program, f'{array_name}[{index}].x_sweep', config.x_sweeps.get(param_name, 0.0))
        tryset(self.entity_update_program, f'{array_name}[{index}].y_sweep', config.y_sweeps.get(param_name, 0.0))
        tryset(self.entity_update_program, f'{array_name}[{index}].cohort_sweep', config.cohort_sweeps.get(param_name, 0.0))

    def _set_multi_load_uniforms(self, multi_load_service):
        """Set uniforms and SSBO for multi-load mode."""
        config_count = multi_load_service.get_config_count()

        # Set multi-load control uniforms (small, not expensive)
        tryset(self.entity_update_program, 'MULTILOAD_COUNT', config_count)
        tryset(self.entity_update_program, 'MULTI_LOAD_CURRENT_PROGRESS', multi_load_service.current_progress)
        tryset(self.entity_update_program, 'MULTI_LOAD_SIMULTANEOUS_CONFIGS', multi_load_service.simultaneous_configs)

        # Set assignment mode and per-config flags
        assignment_mode_int = 1 if multi_load_service.assignment_mode == "Random" else 0
        tryset(self.entity_update_program, 'MULTI_LOAD_ASSIGNMENT_MODE', assignment_mode_int)
        tryset(self.entity_update_program, 'MULTI_LOAD_PER_CONFIG_INITIAL_CONDITIONS', multi_load_service.per_config_initial_conditions)
        tryset(self.entity_update_program, 'MULTI_LOAD_PER_CONFIG_COHORTS', multi_load_service.per_config_cohorts)
        tryset(self.entity_update_program, 'MULTI_LOAD_PER_CONFIG_HAZARD_RATE', multi_load_service.per_config_hazard_rate)

        # Write config data to SSBO only when dirty (expensive operation)
        if multi_load_service.is_ssbo_dirty():
            self._write_multi_load_ssbo(multi_load_service)
            multi_load_service.clear_ssbo_dirty()

    def _write_multi_load_ssbo(self, multi_load_service):
        """Pack config data and write to SSBO."""
        import struct

        config_count = multi_load_service.get_config_count()
        data = bytearray()

        for i in range(config_count):
            config = multi_load_service.get_config(i)
            if config is None:
                # Write zeros for missing configs (10×7 floats + 6 ints + 3 floats = 316 bytes)
                data.extend(bytes(316))
                continue

            # Pack physics parameters (10 PhysicsSetting structs, each 7 floats)
            params = [
                ('axial_force', 'AXIAL_FORCE', -1.0, 1.0),
                ('lateral_force', 'LATERAL_FORCE', -1.0, 1.0),
                ('sensor_gain', 'SENSOR_GAIN', 0.0, 5.0),
                ('mutation_scale', 'MUTATION_SCALE', -0.5, 0.5),
                ('drag', 'DRAG', -1.0, 1.0),
                ('strafe_power', 'STRAFE_POWER', 0.0, 0.5),
                ('sensor_angle', 'SENSOR_ANGLE', -1.0, 1.0),
                ('global_force_mult', 'GLOBAL_FORCE_MULT', 0.0, 2.0),
                ('sensor_distance', 'SENSOR_DISTANCE', 0.0, 4.0),
                ('hazard_rate', 'HAZARD_RATE', 0.0, 0.05),
            ]

            for attr_name, param_name, default_min, default_max in params:
                slider_value = getattr(config, attr_name)
                min_val, max_val = self._get_slider_range(attr_name.replace('_', ' ').title(), default_min, default_max)
                if config.parameter_sweeps_enabled:
                    x_sweep = config.x_sweeps.get(param_name, 0.0)
                    y_sweep = config.y_sweeps.get(param_name, 0.0)
                else:
                    x_sweep = 0
                    y_sweep = 0
                cohort_sweep = config.cohort_sweeps.get(param_name, 0.0)
                jitter = config.jitters.get(param_name, 0.0)
                data.extend(struct.pack('7f', slider_value, min_val, max_val, x_sweep, y_sweep, cohort_sweep, jitter))

            # Pack simulation settings (6 ints)
            data.extend(struct.pack('6i',
                int(config.disable_symmetry),
                int(config.absolute_orientation),
                config.boundary_conditions,
                config.initial_conditions,
                config.num_cohorts,
                int(config.color_by_cohort)
            ))

            # Pack appearance, orientation_mix, and rule seed (3 floats)
            data.extend(struct.pack('3f',
                config.hue_sensitivity,
                config.orientation_mix,
                config.rule_seed
            ))

        # Write config data to SSBO
        self.multi_load_buffer.write(bytes(data))

        # Write rules to separate rule buffer
        rule_data = bytearray()
        for i in range(config_count):
            config = multi_load_service.get_config(i)
            if config is None or config.rule is None:
                # Write zeros for missing rules
                rule_data.extend(bytes(SIZE_OF_RULE_STRUCT))
            else:
                # Write rule as flat float32 array (10 centers * 8 floats = 80 floats)
                rule_data.extend(config.rule.astype(np.float32).tobytes())

        self.multi_load_rule_buffer.write(bytes(rule_data))

    def apply_rule(self, rule: np.ndarray | None) -> None:
        """Apply a rule to the shader."""
        if rule is None:
            set_rule_uniform(self.entity_update_program, np.zeros((10, 8), dtype=np.float32))
        else:
            set_rule_uniform(self.entity_update_program, rule)

    def get_entity_buffer(self) -> moderngl.Buffer:
        """Expose entity buffer for EntityPicker."""
        return self.entities

    def get_rule_buffer(self) -> moderngl.Buffer:
        """Expose rule buffer for rule readback."""
        return self.rule_buffer

    def request_rule_buffer_update(self, entity_id: int) -> None:
        """Request a one-time rule buffer write for the next frame.

        This triggers the expensive rule buffer write (192MB) for exactly one frame,
        allowing subsequent readback of the mutated rule for the specified entity.

        Args:
            entity_id: The entity index to read back after the buffer is written
        """
        self._pending_rule_buffer_update = True
        self._pending_entity_id = entity_id

    def consume_pending_rule_readback(self) -> int | None:
        """Check if a rule readback is ready and consume the pending state.

        Call this AFTER entity_update has run. If a rule buffer update was pending,
        this returns the entity ID to read back and clears the pending state.

        Returns:
            Entity ID to read back, or None if no readback is pending
        """
        if self._pending_rule_buffer_update and self._pending_entity_id is not None:
            entity_id = self._pending_entity_id
            # Clear the pending state - the rule buffer has been written this frame
            self._pending_rule_buffer_update = False
            self._pending_entity_id = None
            return entity_id
        return None

    def update_sliders_from_particle(self, pos: tuple[float, float], cohort: float) -> None:
        """Update all slider values based on effective values at a particle's position/cohort.

        When a particle is clicked and parameter sweeps are active, this calculates what
        the effective parameter values are at that particle's location and updates the
        sliders to show those values.

        Args:
            pos: (x, y) world position of entity in [-1, 1] range
            cohort: Normalized cohort value in [0, 1] range
        """
        # Define all 12 parameters with their state field, slider label, and default ranges
        parameters = [
            ('AXIAL_FORCE', 'Axial Force', -1.0, 1.0),
            ('LATERAL_FORCE', 'Lateral Force', -1.0, 1.0),
            ('SENSOR_GAIN', 'Sensor Gain', 0.0, 5.0),
            ('MUTATION_SCALE', 'Mutation Scale', -0.5, 0.5),
            ('DRAG', 'Drag', -1.0, 1.0),
            ('STRAFE_POWER', 'Strafe Power', 0.0, 0.5),
            ('SENSOR_ANGLE', 'Sensor Angle', -1.0, 1.0),
            ('GLOBAL_FORCE_MULT', 'Global Force Mult', 0.0, 2.0),
            ('SENSOR_DISTANCE', 'Sensor Distance', 0.0, 4.0),
            ('TRAIL_PERSISTENCE', 'Trail Persistence', 0.0, 1.0),
            ('TRAIL_DIFFUSION', 'Trail Diffusion', 0.0, 1.0),
            ('HAZARD_RATE', 'Hazard Rate', 0.0, 0.05),
        ]

        for param_name, slider_label, default_min, default_max in parameters:
            # Get current slider value
            current_value = getattr(self._state, param_name)

            # Get sweep states for this parameter (only if parameter sweeps UI is enabled)
            if self._state.parameter_sweeps_enabled:
                x_sweep = self._state.x_sweeps.get(param_name, 0.0)
                y_sweep = self._state.y_sweeps.get(param_name, 0.0)
                cohort_sweep = self._state.cohort_sweeps.get(param_name, 0.0)
            else:
                x_sweep = 0.0
                y_sweep = 0.0
                cohort_sweep = 0.0

            # Only update if at least one sweep is active
            if x_sweep != 0.0 or y_sweep != 0.0 or cohort_sweep != 0.0:
                # Get min/max range for this parameter
                min_val, max_val = self._get_slider_range(slider_label, default_min, default_max)

                # Calculate effective value at this particle's position/cohort
                effective_value = self.calculate_setting(
                    current_value, min_val, max_val,
                    pos, cohort,
                    x_sweep, y_sweep, cohort_sweep
                )

                # Update the slider value
                setattr(self._state, param_name, effective_value)

    def has_active_cohort_sweep(self) -> bool:
        """Check if any cohort sweep is active."""
        if not self._state.parameter_sweeps_enabled:
            return False
        return any(v != 0.0 for v in self._state.cohort_sweeps.values())

    def has_active_xy_sweep(self) -> bool:
        """Check if any X or Y sweep is active."""
        if not self._state.parameter_sweeps_enabled:
            return False
        has_x = any(v != 0.0 for v in self._state.x_sweeps.values())
        has_y = any(v != 0.0 for v in self._state.y_sweeps.values())
        return has_x or has_y

    def update_sliders_from_position(self, pos: tuple[float, float]) -> None:
        """Update slider values based on position only (no cohort info needed).

        Used when clicking in parameter sweep mode without a cohort sweep active.
        Uses cohort=0.5 as a neutral value.

        Args:
            pos: (x, y) world position in [-1, 1] range
        """
        self.update_sliders_from_particle(pos, cohort=0.5)

    def get_sweep_reticle_position(self) -> tuple[float, float, bool]:
        """Calculate the reticle position based on current slider values and active sweeps.

        The reticle shows where on the screen the current slider values correspond to.
        This is the location where physics doesn't change when toggling sweeps.

        Returns:
            (x, y, visible): UV coordinates (0-1) and whether reticle should be visible.
                             Returns (0.5, 0.5, False) if no X/Y sweeps are active.
        """
        if not self._state.parameter_sweeps_enabled:
            return (0.5, 0.5, False)

        # Find the active X and Y sweep parameters
        x_param = None
        x_sweep_mode = 0.0
        y_param = None
        y_sweep_mode = 0.0

        for param_name in self._state.x_sweeps:
            mode = self._state.x_sweeps.get(param_name, 0.0)
            if mode != 0.0:
                x_param = param_name
                x_sweep_mode = mode
                break

        for param_name in self._state.y_sweeps:
            mode = self._state.y_sweeps.get(param_name, 0.0)
            if mode != 0.0:
                y_param = param_name
                y_sweep_mode = mode
                break

        # If no X or Y sweep is active, don't show reticle
        if x_param is None and y_param is None:
            return (0.5, 0.5, False)

        # Parameter definitions for getting slider ranges
        param_ranges = {
            'AXIAL_FORCE': ('Axial Force', -1.0, 1.0),
            'LATERAL_FORCE': ('Lateral Force', -1.0, 1.0),
            'SENSOR_GAIN': ('Sensor Gain', 0.0, 5.0),
            'MUTATION_SCALE': ('Mutation Scale', -0.5, 0.5),
            'DRAG': ('Drag', -1.0, 1.0),
            'STRAFE_POWER': ('Strafe Power', 0.0, 0.5),
            'SENSOR_ANGLE': ('Sensor Angle', -1.0, 1.0),
            'GLOBAL_FORCE_MULT': ('Global Force Mult', 0.0, 2.0),
            'SENSOR_DISTANCE': ('Sensor Distance', 0.0, 4.0),
            'TRAIL_PERSISTENCE': ('Trail Persistence', 0.0, 1.0),
            'TRAIL_DIFFUSION': ('Trail Diffusion', 0.0, 1.0),
            'HAZARD_RATE': ('Hazard Rate',0.0,0.05)
        }

        # Calculate X position
        if x_param is not None:
            slider_label, default_min, default_max = param_ranges[x_param]
            min_val, max_val = self._get_slider_range(slider_label, default_min, default_max)
            slider_value = getattr(self._state, x_param)
            # Invert the sweep formula: pos_norm = (slider_value - min) / (max - min)
            if max_val != min_val:
                x_norm = (slider_value - min_val) / (max_val - min_val)
            else:
                x_norm = 0.5
            # For inverse sweep, flip the position
            if x_sweep_mode < 0:
                x_norm = 1.0 - x_norm
            reticle_x = x_norm
        else:
            reticle_x = 0.5  # No X sweep - use center

        # Calculate Y position
        if y_param is not None:
            slider_label, default_min, default_max = param_ranges[y_param]
            min_val, max_val = self._get_slider_range(slider_label, default_min, default_max)
            slider_value = getattr(self._state, y_param)
            if max_val != min_val:
                y_norm = (slider_value - min_val) / (max_val - min_val)
            else:
                y_norm = 0.5
            if y_sweep_mode < 0:
                y_norm = 1.0 - y_norm
            reticle_y = y_norm
        else:
            reticle_y = 0.5  # No Y sweep - use center

        return (reticle_x, reticle_y, True)
