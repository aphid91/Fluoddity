"""Command handler: processes one-shot UI commands each frame."""
import random
import time
import numpy as np
from utilities.gl_helpers import readback_rule
from sim import RULE_BUFFER_SIZE
from camera_input import sync_orbit_angles_from_camera
from config_clipboard import ConfigClipboardHandler


class CommandHandler:
    """Processes one-shot commands from UI state.

    Receives references to app components in __init__ and coordinates
    them in response to UI flags (the "one-shot flag" pattern).
    """

    def __init__(self, sim, camera, ui, rule_manager, entity_picker,
                 video_service, config_saver, user_configs_dir,
                 param_lock_service=None, render_spec_service=None,
                 editor_saver=None, simulation_saver=None,
                 recording_controller=None, controller_cam=None, plotting_manager=None,
                 renderer_host=None):
        self.sim = sim
        self.camera = camera
        self.ui = ui
        self.rule_manager = rule_manager
        self.entity_picker = entity_picker
        # Renderer host — used to route entity picking through the OptiX GAS
        # (single-ray pick) when OptiX is the active renderer.
        self.renderer_host = renderer_host
        self.video_service = video_service
        self.config_saver = config_saver
        self.user_configs_dir = user_configs_dir
        self.param_lock_service = param_lock_service
        self.render_spec_service = render_spec_service
        self.editor_saver = editor_saver
        self.simulation_saver = simulation_saver
        self.recording_controller = recording_controller

        self.controller_cam = controller_cam
        self.plotting_manager = plotting_manager

        # Preview state
        # The "remembered original" for file/Load-menu preview: a (config, field_snapshot)
        # tuple captured on preview-enter and reloaded verbatim on preview-leave.
        # None when no file preview is active. Preview NEVER touches the RuleManager
        # undo stack; restore re-applies the cached rule to the GPU directly.
        self._preview_restore = None

        # Config clipboard preview/load/delete handlers (Step 9): owns its own
        # preview state and operates on the UI's ConfigClipboardState.
        self.clipboard_handler = ConfigClipboardHandler(
            sim, rule_manager, config_saver, ui.clipboard_state,
            self._apply_config_with_locks, self._push_and_apply_rule,
            ui.update_physics_defaults,
            param_lock_service=param_lock_service)

        # Deferred entity selection state (waits one frame for rule buffer to be written)
        self._pending_entity_selection = None  # Tuple of (entity_id, entity_pos, entity_cohort) or None

    def _apply_config_with_locks(self, config, ui_state):
        """Apply config with parameter lock snapshot/restore. Returns rule."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(ui_state.sim, ui_state.preferences) if pls else {}
        rule = self.config_saver.apply_config(config, ui_state.sim)
        if pls:
            pls.restore_locked(ui_state.sim, ui_state.preferences, snapshot)
        self.config_applied_this_frame = True
        return rule

    def _push_and_apply_rule(self, rule, ui_state):
        """Push rule to manager and apply to GPU, unless rule lock is active."""
        pls = self.param_lock_service
        if pls and pls.should_block_rule_push():
            return
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)

    def load_full_config(self, config, ui_state, *, json_filepath=None,
                         field_snapshot=None, push_rule=True):
        """Load a whole PhysicsConfig to live state: params + rule.

        The single "load a full config" primitive. Composes the two steps that
        were previously duplicated across fresh-load / clipboard-load / preview:
          1. physics params + appearance + 3D settings (lock-aware) via apply_config
          2. the rule to the GPU

        Args:
            config: PhysicsConfig to apply.
            ui_state: Frame state (sim + preferences), mutated in place.
            json_filepath: Unused (kept for signature stability; the live force/
                strafe field runtime was removed with the drawing mode). Field
                data on disk is preserved but no longer applied.
            field_snapshot: Unused (see json_filepath).
            push_rule: True for real (undoable) loads -> pushes onto RuleManager.
                False for preview -> applies the rule to the GPU directly without
                ever touching the undo stack.
        """
        pls = self.param_lock_service
        rule = self._apply_config_with_locks(config, ui_state)

        if push_rule:
            self._push_and_apply_rule(rule, ui_state)
        else:
            # Preview: apply the rule straight to the GPU, never the undo stack.
            if not (pls and pls.should_block_rule_push()):
                if not (pls and pls.is_locked('rule_seed')):
                    ui_state.sim.rule_seed = config.rule_seed
                self.sim.apply_rule(rule)
        return rule

    @property
    def has_pending_entity_selection(self):
        """Whether there's a pending entity selection waiting for rule readback."""
        return self._pending_entity_selection is not None

    def try_complete_entity_selection(self, ui_state):
        """Try to complete a pending entity selection after rule buffer update.

        Called from simulation_runner after sim.update() on the first physics step.
        Returns True if selection was completed.
        """
        if self._pending_entity_selection is None:
            return False

        pending_entity_id = self.sim.consume_pending_rule_readback()
        if pending_entity_id is None:
            return False

        entity_id, entity_pos, entity_cohort = self._pending_entity_selection
        self._pending_entity_selection = None

        # Read back the rule — cohort maps to a slot in the fixed-size rule buffer
        cohort_slot = int(entity_cohort) % RULE_BUFFER_SIZE
        rule = readback_rule(self.sim.get_rule_buffer(), cohort_slot)
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)
        self.sim.update_sliders_from_particle(entity_pos, entity_cohort)
        print(f"Deferred rule readback complete for entity {entity_id}")
        return True

    def process_commands(self, ui_state):
        """Handle one-shot commands from UI state."""
        self.config_applied_this_frame = False

        # Pick focal entity (N key) — set focal plane to nearest entity depth
        if ui_state.request_pick_focal:
            self._handle_pick_focal(ui_state)

        # Handle world size change
        if ui_state.request_world_size_change:
            self._handle_world_size_change(ui_state)

        # Toggle recording (with delayed start support)
        if ui_state.toggle_recording:
            self.recording_controller.toggle(ui_state)

        # Screenshot request (Shift+P) - set pending flag
        if ui_state.request_screenshot:
            return 'screenshot_pending'

        # Shader reload
        if ui_state.request_reload:
            self.sim.reload()
            if self.rule_manager.has_rules():
                self.sim.apply_rule(self.rule_manager.get_current_rule())
            self.camera.reload()
            if self.plotting_manager is not None:
                self.plotting_manager.reload_shader()
            if self.ui._tracer_interface is not None:
                self.ui._tracer_interface.reload_shaders()
            if getattr(self.ui, '_pathtracer_interface', None) is not None:
                self.ui._pathtracer_interface.reload_shaders()

        # Simple reset (R key)
        if ui_state.request_reset:
            self.sim.reset()

        # Full reset (Z key)
        if ui_state.request_full_reset:
            self._handle_full_reset(ui_state)

        # Randomize mutations (M key)
        if ui_state.request_randomize_mutations:
            self._handle_randomize_mutations(ui_state)

        # Handle mouse clicks
        canvas_aspect_ratio = ui_state.preferences.rendering.canvas_aspect_ratio
        #convert aspect string to tuple of floats
        canvas_aspect_ratio = tuple(float(x) for x in canvas_aspect_ratio.split(":"))
        #convert to ratio
        canvas_aspect_ratio = canvas_aspect_ratio[1]/canvas_aspect_ratio[0]
        self._handle_mouse_clicks(ui_state, canvas_aspect_ratio)

        # Handle config save/load/delete
        self._handle_config_commands(ui_state)

        # Save render spec
        if ui_state.request_save_render_spec:
            self._handle_save_render_spec(ui_state)

        # Preview render spec (destructive apply)
        if ui_state.request_preview_render_spec:
            self._handle_preview_render_spec(ui_state)

        # Editor settings save/load
        if ui_state.request_save_editor:
            self._handle_save_editor(ui_state)
        if ui_state.request_load_editor:
            self._handle_load_editor(ui_state)
        if ui_state.request_reset_ui_settings:
            self._handle_reset_ui_settings(ui_state)

        # Simulation state save/load
        if ui_state.request_save_simulation:
            self._handle_save_simulation(ui_state)
        if ui_state.request_load_simulation:
            self._handle_load_simulation(ui_state)

        # Handle preview commands (file browser)
        self._handle_preview_commands(ui_state)

        # Handle config clipboard commands
        self.clipboard_handler.process(ui_state)

        return None

    def _handle_world_size_change(self, ui_state):
        """Handle entity count / canvas resolution change."""
        self.sim._entity_count = ui_state.preferences.rendering.entity_count
        self.sim.canvas_resolution = ui_state.preferences.rendering.canvas_resolution
        self.sim.setup_simulation_state()
        self.sim.setup_shaders()
        self.entity_picker.update_buffer(self.sim.get_entity_buffer())
        if self.rule_manager.has_rules():
            self.sim.apply_rule(self.rule_manager.get_current_rule())
        self.sim.reset()
        self.ui._last_applied_entity_count = ui_state.preferences.rendering.entity_count
        self.ui._last_applied_canvas_resolution = ui_state.preferences.rendering.canvas_resolution
        print(f"World size changed "
              f"(entity_count: {self.sim.entity_count}, "
              f"canvas: {self.sim.get_canvas_dimensions()[0]}x{self.sim.get_canvas_dimensions()[1]})")

    def _handle_full_reset(self, ui_state):
        """Handle full reset (Z key): reset entities, apply zero rule, randomize, push new state."""
        self.sim.reset()
        zero_rule = np.zeros((10, 12), dtype=np.float32)
        self.sim.apply_rule(zero_rule)
        ui_state.sim.rule_seed = random.random()
        self.rule_manager.push_rule(zero_rule, ui_state.sim.rule_seed)

    def _handle_randomize_mutations(self, ui_state):
        """Handle randomize mutations (M key)."""
        current_rule = self.rule_manager.get_current_rule()
        if current_rule is not None:
            ui_state.sim.rule_seed = random.random()
            self.rule_manager.push_rule(current_rule.copy(), ui_state.sim.rule_seed)
            self.sim.apply_rule(current_rule)

    def _pick_entity_3d(self, ui_state, ray_origin, ray_dir):
        """Pick an entity along a 3D ray.

        When OptiX is the active renderer, cast a real ray into the OptiX GAS
        and select the first entity hit (respects sphere radii + occlusion). On
        an OptiX miss / SDF hit / out-of-range index, fall back to the CPU
        nearest-particle-to-ray picker.

        Returns (entity_id, (pos_x, pos_y), cohort_value, depth).
        """
        if ui_state.camera.optix_enabled and self.renderer_host is not None \
                and self.renderer_host.optix is not None:
            hit = self.renderer_host.optix.pick(ray_origin, ray_dir)
            if hit is not None:
                prim, depth = hit
                if 0 <= prim < self.sim.entity_count:
                    pos, cohort = self.entity_picker.get_entity_by_index(
                        prim, num_cohorts=ui_state.sim.num_cohorts,
                        active_count=self.sim.entity_count)
                    return (prim, pos, cohort, depth)

        # Fallback: CPU nearest-particle-to-ray.
        return self.entity_picker.find_nearest_entity_3d(
            ray_origin, ray_dir,
            num_cohorts=ui_state.sim.num_cohorts, active_count=self.sim.entity_count)

    def _handle_pick_focal(self, ui_state, screen_pos=None):
        """Handle N key: pick nearest entity to mouse, set focal plane and orbit center.

        screen_pos overrides the mouse position (e.g. gamepad D-pad rack-focus
        at the canvas center).
        """
        ray_origin, ray_dir = self.camera.screen_to_ray_3d(screen_pos or ui_state.mouse_pos)
        _entity_id, _entity_pos, _cohort, depth = self._pick_entity_3d(
            ui_state, ray_origin, ray_dir)
        if depth > 0:
            ui_state.camera.focal_plane_depth = depth
            ui_state.camera.orbit_center[:] = ray_origin + ray_dir * depth
            # Sync orbit angles so camera doesn't jump when center changes
            if self.camera.controller_cam is not None:
                sync_orbit_angles_from_camera(ui_state.camera, self.camera.controller_cam)

    def _handle_mouse_clicks(self, ui_state, canvas_aspect_ratio):
        """Handle left/right mouse click behavior based on mode.

        Parameter sweeps take priority over the Select-Particle rule picker:
        while sweeps are enabled, a left-click copies the clicked entity's swept
        slider values (no rule change), so the pick never disturbs the active
        rule. Otherwise, in "Select Particle" mode left-click loads the entity's
        rule and right-click pops the rule stack.
        """
        if ui_state.sim.parameter_sweeps_enabled:
            if ui_state.left_click_this_frame:
                self._handle_sweep_click(ui_state)
            return

        if ui_state.preferences.ui_windows.mouse_mode != "Select Particle":
            return

        if ui_state.left_click_this_frame:
            self._handle_entity_pick(ui_state, canvas_aspect_ratio)
        elif ui_state.right_click_this_frame:
            self._handle_undo(ui_state)

    def _handle_undo(self, ui_state):
        """Pop the current rule off the undo stack (equivalent to right click)."""
        if self.rule_manager.length() > 1:
            prev_rule, prev_seed = self.rule_manager.pop_rule()
            if prev_seed is not None:
                ui_state.sim.rule_seed = prev_seed
            self.sim.apply_rule(prev_rule)

    def handle_gamepad_dpad(self, ui_state, *, pick_pressed, undo_pressed, focus_pressed):
        """D-pad shortcuts, mirroring left-click / right-click / N-key at screen center.

        Up = pick entity at canvas center (left click), Down = undo (right
        click), Left = rack focus at canvas center (hover-center + N).
        """
        if pick_pressed and ui_state.preferences.ui_windows.mouse_mode == "Select Particle" \
                and not ui_state.sim.parameter_sweeps_enabled:
            canvas_aspect_ratio = ui_state.preferences.rendering.canvas_aspect_ratio
            canvas_aspect_ratio = tuple(float(x) for x in canvas_aspect_ratio.split(":"))
            canvas_aspect_ratio = canvas_aspect_ratio[1] / canvas_aspect_ratio[0]
            center = self.camera.get_screen_center()
            self._handle_entity_pick(ui_state, canvas_aspect_ratio, screen_pos=center)

        if undo_pressed and ui_state.preferences.ui_windows.mouse_mode == "Select Particle" \
                and not ui_state.sim.parameter_sweeps_enabled:
            self._handle_undo(ui_state)

        if focus_pressed:
            center = self.camera.get_screen_center()
            self._handle_pick_focal(ui_state, screen_pos=center)

    def _handle_sweep_click(self, ui_state):
        """Left click while parameter sweeps are enabled: copy the clicked
        entity's swept slider values into the sliders (via the entity-picker
        raycast). Does NOT read back or change the entity's rule — sweep mode
        only edits slider values, not behavior.
        """
        ray_origin, ray_dir = self.camera.screen_to_ray_3d(ui_state.mouse_pos)
        entity_id, entity_pos, entity_cohort, _depth = self._pick_entity_3d(
            ui_state, ray_origin, ray_dir)
        if 0 <= entity_id < self.sim.entity_count:
            self.sim.update_sliders_from_particle(entity_pos, entity_cohort)

    def _handle_entity_pick(self, ui_state, canvas_aspect_ratio, screen_pos=None):
        """Handle entity selection via left click in Select Particle mode (3D ray pick).

        screen_pos overrides the mouse position (e.g. gamepad D-pad pick at
        the canvas center).
        """
        ray_origin, ray_dir = self.camera.screen_to_ray_3d(screen_pos or ui_state.mouse_pos)
        entity_id, entity_pos, entity_cohort, _depth = self._pick_entity_3d(
            ui_state, ray_origin, ray_dir)

        if entity_id >= 0 and entity_id < self.sim.entity_count:
            print(f"Entity {entity_id} at pos {entity_pos}, cohort {entity_cohort} - requesting rule buffer update")
            self.sim.request_rule_buffer_update(entity_id)
            self._pending_entity_selection = (entity_id, entity_pos, entity_cohort)
        else:
            print(f"Warning: entity_id {entity_id} out of bounds (max: {self.sim.entity_count - 1})")

    def _handle_config_commands(self, ui_state):
        """Handle config save/load/delete commands.

        The live force/strafe field runtime was removed with the drawing mode,
        so new saves never carry field data (``field_strengths=None``). Field
        data in *old* saves on disk is left untouched — the config/render_spec
        serializers still round-trip ``field_strengths`` for a future step.
        """
        # Config save (Ctrl+C)
        if ui_state.request_save_config:
            current_rule = self.rule_manager.get_current_rule()
            config = self.config_saver.create_config(
                ui_state.sim, current_rule, field_strengths=None)
            config_string = self.config_saver.encode_clipboard(config)
            self.ui.set_clipboard(config_string)
            self.ui.clipboard_state.add(
                config, self.ui.currently_open_project, field_snapshot=None)
            print(f"Config copied to clipboard ({len(config_string)} chars)")

        # Config load (Ctrl+V)
        if ui_state.request_load_config:
            config_string = ui_state.clipboard_text
            if config_string:
                config = self.config_saver.decode_clipboard(config_string)
                if config is not None:
                    rule = self._apply_config_with_locks(config, ui_state)
                    self._push_and_apply_rule(rule, ui_state)
                    print("Config loaded from clipboard")
                else:
                    print("Failed to load config from clipboard")

        # File save (menu)
        if ui_state.request_save_file:
            self._handle_file_save(ui_state)

        # File load (menu)
        if ui_state.request_load_file:
            self._handle_file_load(ui_state)

        # File delete (menu)
        if ui_state.request_delete_file:
            filename = ui_state.delete_filename
            category = ui_state.delete_category
            if filename:
                filepath = self.ui._get_config_path(filename, category)
                if filepath.exists():
                    filepath.unlink()
                    print(f"Config deleted: {filepath}")

    def _handle_file_save(self, ui_state):
        """Handle file save from menu."""
        filename = ui_state.save_filename
        if not filename:
            return

        current_rule = self.rule_manager.get_current_rule()
        config = self.config_saver.create_config(
            ui_state.sim, current_rule, field_strengths=None)
        filepath = self.user_configs_dir / f"{filename}.json"
        self.config_saver.save_to_file(config, filepath)

        print(f"Config saved to {filepath}")
        self.ui.update_physics_defaults(filename)

    def _handle_file_load(self, ui_state):
        """Handle file load (menu click) — always a fresh, undoable load.

        Under the simplified preview model, preview never pushes to the undo
        stack, so a commit is always a normal fresh load. Committing discards the
        remembered preview original and cancels any same-frame clear-preview
        request so it cannot restore over the just-loaded config.
        """
        filename = ui_state.load_filename
        if not filename:
            return

        # Commit: drop the remembered original and suppress a same-frame restore.
        # (_handle_file_load runs before _handle_preview_commands within the frame.)
        self._preview_restore = None
        ui_state.request_clear_preview = False

        filepath = self.ui._get_config_path(filename, ui_state.load_category)
        config = self.config_saver.load_from_file(filepath)
        if config is not None:
            self.load_full_config(
                config, ui_state,
                json_filepath=filepath,
                push_rule=True,
            )
            print(f"Config loaded from {filepath}")
            self.ui.update_physics_defaults(filename)
        else:
            print(f"Failed to load config from {filepath}")

    def _handle_preview_commands(self, ui_state):
        """Handle config preview (hover in Load submenu) and clear preview.

        Model: enter -> cache current config, load preview; change -> load new
        preview (cache untouched); leave -> load cached config. Preview never
        touches the RuleManager undo stack.
        """
        # A "change" (both flags set this frame, e.g. hovering A -> B) must NOT
        # restore-then-reload: that would re-capture the override-mutated live
        # state as the "original". Only a genuine leave (clear with no new
        # preview) restores. On a change we just load the new config over the
        # top; the remembered original stays untouched.
        if ui_state.request_clear_preview and not ui_state.request_preview_config:
            if self._preview_restore is not None:
                config, field_snapshot = self._preview_restore
                self.load_full_config(
                    config, ui_state,
                    field_snapshot=field_snapshot,
                    push_rule=False,
                )
                self._preview_restore = None

        # New / changed preview
        if ui_state.request_preview_config:
            filename = ui_state.preview_filename
            category = ui_state.preview_category
            if filename:
                filepath = self.ui._get_config_path(filename, category)
                config = self.config_saver.load_from_file(filepath)
                if config and config.rule is not None:
                    # Cache the current live config on first preview entry only.
                    if self._preview_restore is None:
                        self._capture_preview_restore(ui_state)
                    self.load_full_config(
                        config, ui_state,
                        json_filepath=filepath,
                        push_rule=False,
                    )

    def _capture_preview_restore(self, ui_state):
        """Snapshot the current live state as the preview's remembered original."""
        current_rule = self.rule_manager.get_current_rule()
        config = self.config_saver.create_config(
            ui_state.sim, current_rule, field_strengths=None)
        self._preview_restore = (config, None)

    def _sync_tracer_to_preferences(self, ui_state):
        """Sync live TracerInterface values into PreferencesState.

        The tracer UI modifies TracerInterface directly, so PreferencesState
        fields are stale until this is called. Needed before any snapshot that
        reads preferences (render spec save, preferences save, etc.).
        """
        ti = self.ui._tracer_interface
        if ti is None:
            return
        p = ui_state.preferences
        p.tracer.sdf_enabled = ti.sdf_enabled
        p.tracer.colored_extinction = ti.colored_extinction
        p.tracer.extinction_rgb = list(ti.extinction_rgb)
        p.tracer.albedo_saturation = ti.albedo_saturation
        p.tracer.albedo_brightness = ti.albedo_brightness
        p.tracer.density_scale = ti.density_scale
        p.tracer.hg_g = ti.hg_g
        p.tracer.emission_strength = ti.emission_strength
        # Sun/sky (LightingPrefs) and rt-mode/capture-spp/resolution/firefly
        # (shared RenderingPrefs) are edited directly on their slices — nothing
        # to sync back from ti.
        p.tracer.max_bounces = ti.max_bounces
        p.tracer.density_resolution_log2 = ti.density_resolution_log2
        p.tracer.color_resolution_log2 = ti.color_resolution_log2
        p.tracer.majorant_resolution_log2 = ti.majorant_resolution_log2

    def _handle_save_render_spec(self, ui_state):
        """Capture current state and save as a render spec to disk."""
        if not self.render_spec_service:
            print("RenderSpecService not available")
            return
        name = ui_state.save_render_spec_name or "render"
        # Sync tracer settings into preferences before capturing
        self._sync_tracer_to_preferences(ui_state)
        try:
            spec, gpu_buffers = self.render_spec_service.capture_current_state(
                self.sim, self.camera, self.controller_cam, ui_state,
                self.config_saver, self.rule_manager,
                name
            )
            saved_path = self.render_spec_service.save_to_disk(spec, gpu_buffers)
            print(f"Render spec saved: {saved_path}")
            self.ui._render_spec_saved_time = time.time()
        except Exception as e:
            print(f"Failed to save render spec: {e}")

    def _handle_preview_render_spec(self, ui_state):
        """Load a render spec from disk and destructively apply it."""
        if not self.render_spec_service:
            print("RenderSpecService not available")
            return
        from pathlib import Path
        dir_path = Path(ui_state.preview_render_spec_path)
        if not dir_path.exists():
            print(f"Render spec not found: {dir_path}")
            return
        try:
            spec = self.render_spec_service.load_metadata(dir_path)
            if spec is None:
                return
            gpu_buffers = self.render_spec_service.load_gpu_buffers(dir_path)
            if gpu_buffers is None:
                return
            world_size_changed = self.render_spec_service.apply_state(
                spec, gpu_buffers,
                self.sim, self.camera, self.controller_cam, ui_state,
                self.config_saver, self.rule_manager,
            )
            if world_size_changed:
                self.entity_picker.update_buffer(self.sim.get_entity_buffer())
                self.ui._last_applied_entity_count = ui_state.preferences.rendering.entity_count
                self.ui._last_applied_canvas_resolution = ui_state.preferences.rendering.canvas_resolution
            # Re-sync tracer interface if it exists (preferences were updated
            # but the live TracerInterface still has stale values)
            if self.ui._tracer_interface is not None:
                self.ui._apply_tracer_preferences(self.ui._tracer_interface)
            # Pause simulation after preview
            ui_state.sim.going = False
            print(f"Previewing render spec: {spec.display_name}")
        except Exception as e:
            print(f"Failed to preview render spec: {e}")

    # --- Editor settings save/load ---------------------------------------

    def _handle_save_editor(self, ui_state):
        """Save non-physics editor state (preferences + imgui layout) to disk."""
        if not self.editor_saver:
            print("EditorSaver not available")
            return
        name = ui_state.save_editor_name or "editor"
        # Sync tracer settings into preferences before capturing.
        self._sync_tracer_to_preferences(ui_state)
        try:
            save = self.editor_saver.create_save(ui_state.preferences)
            path = self.editor_saver.save_to_file(save, name=name)
            print(f"Editor settings saved: {path}")
        except Exception as e:
            print(f"Failed to save editor settings: {e}")

    def _handle_load_editor(self, ui_state):
        """Load non-physics editor state from disk and apply it (in place)."""
        if not self.editor_saver:
            print("EditorSaver not available")
            return
        from pathlib import Path
        path = Path(ui_state.load_editor_path)
        try:
            save = self.editor_saver.load_from_file(path)
            if save is None:
                return
            self.editor_saver.apply_save(save, ui_state.preferences)
            # Re-sync the live 3D camera from the newly-loaded prefs, matching
            # the reset/startup path (stereogram, DOF, orbit, etc. otherwise
            # stay stale until restart).
            self._sync_camera_from_prefs(ui_state)
            # Re-sync the live TracerInterface with the newly-loaded preferences.
            if self.ui._tracer_interface is not None:
                self.ui._apply_tracer_preferences(self.ui._tracer_interface)
            print(f"Editor settings loaded: {path.name}")
        except Exception as e:
            print(f"Failed to load editor settings: {e}")

    def _sync_camera_from_prefs(self, ui_state):
        """Push camera-affecting preferences into the live CameraState.

        Shared by the editor-load and reset paths so the live camera reflects
        newly-applied 3D camera prefs (fov, DOF, orbit, stereogram) without an
        app restart.
        """
        cam = ui_state.camera
        p = ui_state.preferences
        cam.fov = p.camera3d.fov
        cam.aperture = p.camera3d.aperture
        cam.focal_plane_depth = p.camera3d.focal_plane_depth
        cam.move_speed = p.camera3d.move_speed
        cam.rotate_speed = p.camera3d.rotate_speed
        cam.orbit_center[:] = p.camera3d.orbit_center
        cam.orbit_rate = p.camera3d.orbit_rate
        cam.stereogram = p.camera3d.stereogram
        cam.eye_offset = p.camera3d.eye_offset
        cam.stereo_toe_in = p.camera3d.stereo_toe_in

    def _handle_reset_ui_settings(self, ui_state):
        """Reset all UI settings to factory defaults, then overlay the project's
        __Default_Editor save (prefs + docking layout) if it exists.

        Mirrors _handle_load_editor: mutates the live PreferencesState in place
        (identity preserved so held references see the update) and restores the
        imgui docking layout. Falls back cleanly to factory defaults when the
        default editor save is absent.
        """
        from state.preferences_state import PreferencesState, copy_preferences_into
        # Factory defaults first (identity-preserving), then overlay the default
        # editor save (no-op if the file isn't present).
        copy_preferences_into(ui_state.preferences, PreferencesState())
        if self.editor_saver:
            self.editor_saver.apply_default(ui_state.preferences)
        # Re-sync the live 3D camera from the (possibly overlaid) prefs, matching
        # the startup path so camera-affecting prefs take effect.
        self._sync_camera_from_prefs(ui_state)
        p = ui_state.preferences
        ui_state.camera.optix_enabled = (p.rendering.renderer == 1)
        # Re-sync the live TracerInterface with the reset preferences.
        if self.ui._tracer_interface is not None:
            self.ui._apply_tracer_preferences(self.ui._tracer_interface)
        print("UI settings reset to defaults")

    # --- Simulation state save/load --------------------------------------

    def _handle_save_simulation(self, ui_state):
        """Save the entity + canvas GPU buffers to disk."""
        if not self.simulation_saver:
            print("SimulationSaver not available")
            return
        name = ui_state.save_simulation_name or "simulation"
        try:
            buffers = self.simulation_saver.read_buffers(self.sim)
            sim_metadata = self.simulation_saver.sim_metadata(self.sim)
            path = self.simulation_saver.save_to_disk(buffers, sim_metadata, name=name)
            print(f"Simulation state saved: {path}")
        except Exception as e:
            print(f"Failed to save simulation state: {e}")

    def _handle_load_simulation(self, ui_state):
        """Load entity + canvas GPU buffers from disk and write them into the sim."""
        if not self.simulation_saver:
            print("SimulationSaver not available")
            return
        from pathlib import Path
        dir_path = Path(ui_state.load_simulation_path)
        if not dir_path.exists():
            print(f"Simulation state not found: {dir_path}")
            return
        try:
            loaded = self.simulation_saver.load_from_disk(dir_path)
            if loaded is None:
                return
            buffers, sim_metadata = loaded
            sim_metadata = dict(sim_metadata)

            # Particle count + canvas resolution are Editor-owned prefs, but a
            # simulation dump only makes sense at the world size it was captured
            # at. So the SAVE wins: sync its entity_count/canvas_resolution into
            # RenderingPrefs (warning on any change), then let write_buffers
            # reallocate the GPU buffers to match. Fall back to the live values
            # if the save predates canvas_resolution in its metadata.
            r = ui_state.preferences.rendering
            saved_entity_count = sim_metadata.get('entity_count', r.entity_count)
            saved_canvas_res = sim_metadata.get('canvas_resolution', r.canvas_resolution)
            if saved_entity_count != r.entity_count:
                print(f"[SimulationLoad] Particle count changed by load: "
                      f"{r.entity_count} -> {saved_entity_count}")
                r.entity_count = saved_entity_count
            if saved_canvas_res != r.canvas_resolution:
                print(f"[SimulationLoad] Canvas resolution changed by load: "
                      f"{r.canvas_resolution} -> {saved_canvas_res}")
                r.canvas_resolution = saved_canvas_res
            # write_buffers reads entity_count/canvas_resolution from this dict.
            sim_metadata['entity_count'] = saved_entity_count
            sim_metadata['canvas_resolution'] = saved_canvas_res

            rule = self.rule_manager.get_current_rule()
            world_size_changed = self.simulation_saver.write_buffers(
                self.sim, buffers, sim_metadata, rule=rule)
            if world_size_changed:
                self.entity_picker.update_buffer(self.sim.get_entity_buffer())
                self.ui._last_applied_entity_count = r.entity_count
                self.ui._last_applied_canvas_resolution = r.canvas_resolution
            print(f"Simulation state loaded: {dir_path.name}")
        except Exception as e:
            print(f"Failed to load simulation state: {e}")
