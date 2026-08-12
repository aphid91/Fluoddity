"""Simulation runner: physics stepping, frame assembly, and video recording."""
import glfw
import numpy as np


class SimulationRunner:
    """Runs physics simulation steps with frame assembly and video recording.

    Handles both motion-blur (temporal accumulation) and non-motion-blur paths,
    deduplicating the shared physics stepping and frame assembly logic.
    """

    def __init__(self, sim, camera, video_service, command_handler, window,
                 advanced_drawing_processor=None, controller_cam=None):
        self.sim = sim
        self.camera = camera
        self.video_service = video_service
        self.command_handler = command_handler
        self.window = window
        self.advanced_drawing_processor = advanced_drawing_processor
        self.controller_cam = controller_cam

        # Mouse tracking for draw trail mode
        self.prev_mouse_tex_coords = (0.0, 0.0)

    def run_simulation_frame(self, ui_state, sweep_mode, sweep_reticle_pos,
                              sweep_reticle_visible, screen_aspect,
                              watercolor_mode=False, tiling_mode=False,
                              screenshot_in_progress=False):
        """Run simulation step(s) with frame assembly and video recording."""
        self._screenshot_in_progress = screenshot_in_progress
        self.camera.watercolor_mode = watercolor_mode
        speedmult = ui_state.preferences.speedmult
        motion_blur = ui_state.preferences.motion_blur

        # Calculate mouse screen coordinates for draw overlay
        width, height = glfw.get_framebuffer_size(self.window)
        mouse_x_norm = ui_state.mouse_pos[0] / width if width > 0 else 0.5
        mouse_y_norm = ui_state.mouse_pos[1] / height if height > 0 else 0.5
        mouse_screen_coords = (mouse_x_norm, mouse_y_norm)

        # Compute view bounds for tiling mode
        view_min, view_max = self._compute_view_bounds(tiling_mode, screen_aspect)

        # Calculate draw mode parameters
        draw_mode, mouse_tex_coords, draw_power_value = self._compute_draw_params(
            ui_state, tiling_mode
        )

        # Compute erase mode (right-click drag in Draw Trail mode)
        erase_mode = (draw_mode and ui_state.mouse_right_held
                      and not ui_state.mouse_left_held)

        # Advanced Drawing: force/strafe field update (once per render frame, not per physics frame)
        if self.advanced_drawing_processor is not None:
            adv_prefs = ui_state.preferences

            if adv_prefs.advanced_drawing_enabled and adv_prefs.shader_driven_field:
                # Shader-driven field: run override shader every render frame
                cam = self.controller_cam
                self.advanced_drawing_processor.process_override(
                    canvas_width=self.sim.can.size[0],
                    canvas_height=self.sim.can.size[1],
                    shader_name=adv_prefs.field_override_shader,
                    frame_count=self.sim.frame_count,
                    mouse_pos=mouse_tex_coords,
                    prev_mouse_pos=self.prev_mouse_tex_coords,
                    draw_size=adv_prefs.draw_size,
                    draw_power=adv_prefs.draw_power,
                    brush_mode=adv_prefs.brush_mode,
                    fixed_direction_heading=adv_prefs.fixed_direction_heading,
                    tiling_mode=tiling_mode,
                    camera_pos=tuple(cam.pos) if cam else (0.0, 0.0, 0.0),
                    camera_dir=tuple(cam.dir) if cam else (0.0, 0.0, 1.0),
                )
            elif adv_prefs.advanced_drawing_enabled and (
                adv_prefs.advanced_draw_force_field or adv_prefs.advanced_draw_strafe_field
            ):
                # Normal drawing mode: draw/erase to field texture
                field_draw_active = (draw_mode and ui_state.mouse_left_held
                                     and draw_power_value > 0.0)
                # Erase scope: only erase fields that are checked as draw targets
                field_erase = (erase_mode and (
                    adv_prefs.advanced_draw_force_field or adv_prefs.advanced_draw_strafe_field
                ))
                self.advanced_drawing_processor.process(
                    canvas_width=self.sim.can.size[0],
                    canvas_height=self.sim.can.size[1],
                    draw_mode=field_draw_active,
                    mouse_pos=mouse_tex_coords,
                    prev_mouse_pos=self.prev_mouse_tex_coords,
                    draw_size=adv_prefs.draw_size,
                    draw_power=adv_prefs.draw_power,
                    brush_mode=adv_prefs.brush_mode,
                    fixed_direction_heading=adv_prefs.fixed_direction_heading,
                    force_field_active=adv_prefs.advanced_draw_force_field,
                    strafe_field_active=adv_prefs.advanced_draw_strafe_field,
                    tiling_mode=tiling_mode,
                    erase_mode=field_erase,
                    fill_mode=ui_state.request_fill_operation,
                    fill_direction_type=ui_state.fill_direction_type,
                )

            # Handle selective field clear requests
            if ui_state.request_clear_force_field:
                self.advanced_drawing_processor.clear_force_field()
            if ui_state.request_clear_strafe_field:
                self.advanced_drawing_processor.clear_strafe_field()

        # Handle clear canvas request
        if ui_state.request_clear_canvas:
            self.sim.clear_canvas()

        # Handle clear canvas + fields request
        if ui_state.request_clear_canvas_and_fields:
            self.sim.clear_canvas()
            self.advanced_drawing_processor.clear_fields()

        # Build shared frame assembly kwargs (used by both paths)
        assemble_kwargs = self._build_assemble_kwargs(
            ui_state, sweep_mode, sweep_reticle_pos, sweep_reticle_visible,
            screen_aspect, mouse_screen_coords, tiling_mode, view_min, view_max
        )

        if motion_blur:
            self._run_with_motion_blur(
                ui_state, speedmult, draw_mode, mouse_tex_coords, draw_power_value,
                tiling_mode, assemble_kwargs, erase_mode=erase_mode
            )
        else:
            self._run_without_motion_blur(
                ui_state, speedmult, draw_mode, mouse_tex_coords, draw_power_value,
                tiling_mode, assemble_kwargs, erase_mode=erase_mode
            )

        # Update previous mouse position for next frame
        if draw_mode or erase_mode:
            self.prev_mouse_tex_coords = mouse_tex_coords

    def _compute_view_bounds(self, tiling_mode, screen_aspect):
        """Compute view bounds for tiling mode.

        Delegates to camera.compute_tiling_view_bounds() which properly inverts
        the vertex shader transform accounting for both canvas and window aspect.
        """
        if not tiling_mode:
            return (0.0, 0.0), (0.0, 0.0)
        view_min, view_max = self.camera.compute_tiling_view_bounds()
        return tuple(view_min), tuple(view_max)

    def _compute_draw_params(self, ui_state, tiling_mode):
        """Calculate draw mode parameters."""
        # Disable trail drawing when parameter sweeps are active
        draw_mode = (ui_state.preferences.mouse_mode == "Draw Trail" and
                     not ui_state.sim.parameter_sweeps_enabled)
        mouse_tex_coords = (0.0, 0.0)
        draw_power_value = 0.0

        if draw_mode:
            mouse_tex_coords = self.camera.screen_to_tex(
                ui_state.mouse_pos, self.sim.can.size
            )
            if tiling_mode:
                mouse_tex_coords = (
                    np.fmod(mouse_tex_coords[0] + 10.0, 1.0),
                    np.fmod(mouse_tex_coords[1] + 10.0, 1.0)
                )
            if ui_state.mouse_left_held or ui_state.request_fill_operation:
                draw_power_value = ui_state.preferences.draw_power

        return draw_mode, mouse_tex_coords, draw_power_value

    def _get_emboss_params(self, ui_state):
        """Get emboss texture and effective intensity from ui_state."""
        emboss_mode = ui_state.sim.emboss_mode
        if emboss_mode == 1:
            emboss_tex = self.sim.can
        elif emboss_mode == 2:
            emboss_tex = self.sim.brush_tex
        else:
            emboss_tex = None
        effective_emboss_intensity = 0.0 if emboss_mode == 0 else ui_state.sim.emboss_intensity
        return emboss_tex, emboss_mode, effective_emboss_intensity

    def _get_trail_draw_radius(self, ui_state):
        """Calculate trail draw radius (0 when recording, screenshotting, sweeping, or not in Draw Trail mode)."""
        if ui_state.preferences.mouse_mode != "Draw Trail":
            return 0
        if self.video_service.is_active():
            return 0
        if self._screenshot_in_progress:
            return 0
        if ui_state.sim.parameter_sweeps_enabled:
            return 0
        return ui_state.preferences.draw_size

    def _build_assemble_kwargs(self, ui_state, sweep_mode, sweep_reticle_pos,
                                sweep_reticle_visible, screen_aspect,
                                mouse_screen_coords, tiling_mode, view_min, view_max):
        """Build the kwargs dict for frame_assembler.assemble_frame().

        These are shared between motion-blur and non-motion-blur paths.
        Only total_samples and current_sample_index differ between the two.
        """
        emboss_tex, emboss_mode, effective_emboss_intensity = self._get_emboss_params(ui_state)

        adv_prefs = ui_state.preferences
        advanced_active = adv_prefs.advanced_drawing_enabled

        return dict(
            view_mode=ui_state.sim.current_view_option,
            sweep_mode=sweep_mode,
            sweep_reticle_pos=sweep_reticle_pos,
            sweep_reticle_visible=sweep_reticle_visible,
            screen_aspect=screen_aspect,
            brightness=self.camera.BRIGHTNESS,
            exposure=ui_state.preferences.exposure,
            ink_weight=ui_state.sim.ink_weight,
            watercolor_mode=ui_state.sim.watercolor_mode,
            emboss_tex=emboss_tex,
            camera_position=tuple(self.camera.position),
            camera_zoom=self.camera.zoom,
            emboss_intensity=effective_emboss_intensity,
            emboss_smoothness=ui_state.sim.emboss_smoothness,
            trail_draw_radius=self._get_trail_draw_radius(ui_state),
            mouse_screen_coords=mouse_screen_coords,
            tiling_mode=tiling_mode,
            view_min=tuple(view_min),
            view_max=tuple(view_max),
            tiling_scale=self.camera.compute_tiling_scale(),
            canvas_resolution=self.sim.get_canvas_dimensions(),
            tonemap_softness=ui_state.preferences.tonemap_softness,
            brush_mode=adv_prefs.brush_mode if advanced_active else 0,
            fixed_direction_heading=adv_prefs.fixed_direction_heading if advanced_active else 0.0,
            field_texture=(self.advanced_drawing_processor.field_texture
                           if self.advanced_drawing_processor is not None else None),
            advanced_drawing_resources_initialized=(
                self.advanced_drawing_processor is not None
                and self.advanced_drawing_processor.field_texture is not None),
            force_field_checked=adv_prefs.advanced_draw_force_field if advanced_active else False,
            strafe_field_checked=adv_prefs.advanced_draw_strafe_field if advanced_active else False,
            draw_target_overlay_opacity=adv_prefs.draw_target_overlay_opacity if advanced_active else 0.0,
        )

    def _run_physics_step(self, ui_state, draw_mode, mouse_tex_coords,
                           draw_power_value, tiling_mode, step_index,
                           erase_mode=False):
        """Run a single physics step and handle deferred entity selection.

        Args:
            step_index: Current step within the frame (0-based).
                        Entity selection only checked on step 0.
            erase_mode: Whether right-click eraser is active.
        """
        adv_prefs = ui_state.preferences
        advanced_active = adv_prefs.advanced_drawing_enabled

        # Brush mode and heading (only apply when advanced drawing is open)
        brush_mode = adv_prefs.brush_mode if advanced_active else 0
        fixed_heading = adv_prefs.fixed_direction_heading if advanced_active else 0.0

        # Canvas draw active: when advanced drawing is closed, canvas always gets drawn.
        # When open, only if canvas checkbox is checked.
        canvas_draw_active = (not advanced_active) or adv_prefs.advanced_draw_canvas

        # Canvas erase scope: erase canvas if it's a draw target
        canvas_erase = erase_mode and canvas_draw_active

        # Fill mode for canvas: only on first physics step to avoid repeating
        canvas_fill = (ui_state.request_fill_operation and canvas_draw_active
                       and step_index == 0)

        # Only send draw_power if canvas is actually a draw target
        effective_draw_power = draw_power_value if canvas_draw_active else 0.0

        self.sim.update(
            self.camera.ctx,
            draw_mode=draw_mode,
            mouse_pos=mouse_tex_coords,
            prev_mouse_pos=self.prev_mouse_tex_coords,
            draw_size=ui_state.preferences.draw_size,
            draw_power=effective_draw_power,
            multi_load_service=(
                self.command_handler.multi_load_service
                if ui_state.multi_load.multi_load_enabled else None
            ),
            is_preview_active=self.command_handler.preview_rule_active,
            tiling_mode=tiling_mode,
            strong_determinism=ui_state.preferences.strong_determinism,
            brush_mode=brush_mode,
            fixed_direction_heading=fixed_heading,
            erase_mode=canvas_erase,
            fill_mode=canvas_fill,
            fill_direction_type=ui_state.fill_direction_type,
            canvas_draw_active=canvas_draw_active,
            field_texture = self.advanced_drawing_processor.field_texture,
            force_field_strength=adv_prefs.force_field_strength,
            strafe_field_strength=adv_prefs.strafe_field_strength,
        )

        # Check for deferred entity selection only on first physics step
        if step_index == 0 and self.command_handler.has_pending_entity_selection:
            self.command_handler.try_complete_entity_selection(ui_state)

    def _process_assembled_frame(self, assembled_tex, ui_state):
        """Handle a completed assembled frame: store it and feed to video recorder."""
        if assembled_tex is None:
            return
        if ui_state.preferences.bloom_enabled and not ui_state.sim.watercolor_mode:
            assembled_tex = self.camera.apply_bloom(
                assembled_tex,
                ui_state.preferences.bloom_threshold,
                ui_state.preferences.bloom_intensity,
                ui_state.preferences.bloom_radius,
                tonemap_softness=ui_state.preferences.tonemap_softness,
            )
        self.camera.assembled_texture = assembled_tex
        if self.video_service.is_active():
            self.video_service.process_frame(
                self.camera.ctx,
                assembled_tex,
                ui_state.preferences.max_frames,
                ui_state.preferences.supersample_k,
                ui_state.preferences.filename_prefix
            )

    def _run_with_motion_blur(self, ui_state, speedmult, draw_mode,
                               mouse_tex_coords, draw_power_value,
                               tiling_mode, assemble_kwargs, erase_mode=False):
        """Motion blur path: temporal accumulation with multiple render calls."""
        motion_blur_render_cadence = ui_state.preferences.blur_quality
        total_render_samples = (speedmult + motion_blur_render_cadence - 1) // motion_blur_render_cadence
        render_sample_index = 0

        for step in range(speedmult):
            self._run_physics_step(
                ui_state, draw_mode, mouse_tex_coords, draw_power_value,
                tiling_mode, step, erase_mode=erase_mode
            )

            # Only render on frames matching the blur quality cadence
            if step % motion_blur_render_cadence != 0:
                continue

            raw_view_tex = self.camera.generate_view_texture(tiling_mode=tiling_mode)

            assembled_tex = self.camera.frame_assembler.assemble_frame(
                raw_view_tex,
                total_samples=total_render_samples,
                current_sample_index=render_sample_index,
                **assemble_kwargs
            )
            render_sample_index += 1

            self._process_assembled_frame(assembled_tex, ui_state)

    def _run_without_motion_blur(self, ui_state, speedmult, draw_mode,
                                  mouse_tex_coords, draw_power_value,
                                  tiling_mode, assemble_kwargs, erase_mode=False):
        """Non-motion-blur path: multiple physics steps, single render call."""
        for step in range(speedmult):
            self._run_physics_step(
                ui_state, draw_mode, mouse_tex_coords, draw_power_value,
                tiling_mode, step, erase_mode=erase_mode
            )

        raw_view_tex = self.camera.generate_view_texture(tiling_mode=tiling_mode)

        assembled_tex = self.camera.frame_assembler.assemble_frame(
            raw_view_tex,
            total_samples=1,
            current_sample_index=0,
            **assemble_kwargs
        )

        self._process_assembled_frame(assembled_tex, ui_state)
