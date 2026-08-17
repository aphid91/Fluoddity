"""Command handler: processes one-shot UI commands each frame."""
import random
import numpy as np
from utilities.gl_helpers import readback_rule


class CommandHandler:
    """Processes one-shot commands from UI state.

    Receives references to app components in __init__ and coordinates
    them in response to UI flags (the "one-shot flag" pattern).
    """

    def __init__(self, sim, camera, ui, rule_manager, entity_picker,
                 video_service, config_saver, multi_load_service, user_configs_dir,
                 field_handler=None, param_lock_service=None,
                 streamline_service=None, audio_service=None,
                 smoothed_field=None):
        self.sim = sim
        self.camera = camera
        self.ui = ui
        self.rule_manager = rule_manager
        self.entity_picker = entity_picker
        self.video_service = video_service
        self.config_saver = config_saver
        self.multi_load_service = multi_load_service
        self.user_configs_dir = user_configs_dir
        self.field_handler = field_handler
        self.param_lock_service = param_lock_service
        self.streamline_service = streamline_service
        self.audio_service = audio_service
        self.smoothed_field = smoothed_field

        # Preview state
        self.preview_rule_active = False  # File->load preview
        self._preview_rule_was_pushed = False  # Whether we actually pushed a rule (vs blocked by lock)
        self.clipboard_preview_active = False  # Config clipboard preview
        self._clipboard_rule_was_pushed = False  # Whether clipboard preview actually pushed a rule
        self._clipboard_cached_config = None  # Full config saved before clipboard preview

        # Video pending state (waiting for scheduled start frame)
        self.video_pending = False
        self.video_scheduled_start_frame = 0

        # Deferred entity selection state (waits one frame for rule buffer to be written)
        self._pending_entity_selection = None  # Tuple of (entity_id, entity_pos, entity_cohort) or None

    def _apply_config_with_locks(self, config, ui_state, watercolor_override=None):
        """Apply config with parameter lock snapshot/restore. Returns rule."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(ui_state.sim, ui_state.preferences) if pls else {}
        rule = self.config_saver.apply_config(config, ui_state.sim, watercolor_override)
        if pls:
            pls.restore_locked(ui_state.sim, ui_state.preferences, snapshot)
        return rule

    def _push_and_apply_rule(self, rule, ui_state):
        """Push rule to manager and apply to GPU, unless rule lock is active."""
        pls = self.param_lock_service
        if pls and pls.should_block_rule_push():
            return
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)

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

        # Read back the rule (buffer was just written by entity_update)
        rule = readback_rule(self.sim.get_rule_buffer(), entity_id)
        self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        self.sim.apply_rule(rule)
        self.sim.update_sliders_from_particle(entity_pos, entity_cohort)
        print(f"Deferred rule readback complete for entity {entity_id}")
        return True

    def process_commands(self, ui_state, tiling_mode):
        """Handle one-shot commands from UI state."""

        # Handle world size change
        if ui_state.request_world_size_change:
            self._handle_world_size_change(ui_state)

        # Toggle recording (with delayed start support)
        if ui_state.toggle_recording:
            self._handle_toggle_recording(ui_state)

        # Screenshot request (Shift+P) - set pending flag
        if ui_state.request_screenshot:
            return 'screenshot_pending'

        # Shader reload
        if ui_state.request_reload:
            self.sim.reload()
            if self.rule_manager.has_rules():
                self.sim.apply_rule(self.rule_manager.get_current_rule())
            self.camera.reload()
            if self.field_handler and self.field_handler.adv_draw:
                self.field_handler.adv_draw.reload()
            if self.streamline_service:
                self.streamline_service.reload()
            if self.audio_service:
                self.audio_service.reload()
            if self.smoothed_field:
                self.smoothed_field.reload()

        # Simple reset (R key)
        if ui_state.request_reset:
            self.sim.reset()
            # The averaged history describes a canvas that no longer exists,
            # so reseed from the new one rather than blending across the
            # discontinuity.
            if self.smoothed_field:
                self.smoothed_field.request_seed()

        # Full reset (Z key)
        if ui_state.request_full_reset:
            self._handle_full_reset(ui_state)

        # Randomize mutations (M key)
        if ui_state.request_randomize_mutations:
            self._handle_randomize_mutations(ui_state)

        # Handle sweep preview restore
        restored_sweep_preview = self._handle_sweep_preview_restore(ui_state)

        # Handle mouse clicks
        if not restored_sweep_preview:
            canvas_aspect_ratio = ui_state.preferences.canvas_aspect_ratio
            #convert aspect string to tuple of floats
            canvas_aspect_ratio = tuple(float(x) for x in canvas_aspect_ratio.split(":"))
            #convert to ratio
            canvas_aspect_ratio = canvas_aspect_ratio[1]/canvas_aspect_ratio[0]
            self._handle_mouse_clicks(ui_state, tiling_mode, canvas_aspect_ratio)

        # Handle config save/load/delete
        self._handle_config_commands(ui_state)

        # Handle field image load requests
        if ui_state.request_load_force_field_image and ui_state.field_load_image_path:
            if self.field_handler:
                self.field_handler.load_field_from_image(ui_state.field_load_image_path, "force")
        if ui_state.request_load_strafe_field_image and ui_state.field_load_image_path:
            if self.field_handler:
                self.field_handler.load_field_from_image(ui_state.field_load_image_path, "strafe")

        # Handle preview commands (file browser)
        self._handle_preview_commands(ui_state)

        # Handle config clipboard commands
        self._handle_clipboard_commands(ui_state)

        return None

    def _handle_world_size_change(self, ui_state):
        """Handle world size change request (also handles aspect ratio changes)."""
        self.sim.world_size = ui_state.preferences.world_size
        self.sim.canvas_aspect_ratio = ui_state.preferences.canvas_aspect_ratio
        self.sim.setup_simulation_state()
        self.sim.setup_shaders()
        self.entity_picker.update_buffer(self.sim.get_entity_buffer())
        if self.rule_manager.has_rules():
            self.sim.apply_rule(self.rule_manager.get_current_rule())
        self.sim.reset()
        # Reinitialize field texture at new canvas dimensions (if it exists)
        if self.field_handler and self.field_handler._has_field_tex:
            canvas_dim_x, canvas_dim_y = self.sim.get_canvas_dimensions()
            self.field_handler.adv_draw.ensure_initialized(canvas_dim_x, canvas_dim_y)
        self.ui._last_applied_world_size = ui_state.preferences.world_size
        print(f"World size changed to {self.sim.world_size} "
              f"(entity_count: {self.sim.entity_count}, "
              f"canvas: {self.sim.get_canvas_dimensions()[0]}x{self.sim.get_canvas_dimensions()[1]})")

    def _handle_toggle_recording(self, ui_state):
        """Handle video recording toggle with delayed start support."""
        if self.video_pending:
            self.video_pending = False
            self.video_scheduled_start_frame = 0
        elif self.video_service.is_active():
            self.video_service.stop()
        else:
            video_end_frame = ui_state.preferences.video_end_frame
            video_simulation_frames = ui_state.preferences.max_frames * ui_state.preferences.motion_blur_samples
            scheduled_start_frame = video_end_frame - video_simulation_frames

            if video_end_frame == 0 or scheduled_start_frame <= self.sim.frame_count:
                self.video_service.start()
            else:
                self.video_pending = True
                self.video_scheduled_start_frame = scheduled_start_frame

    def _handle_full_reset(self, ui_state):
        """Handle full reset (Z key): reset entities, apply zero rule, randomize, push new state."""
        self.sim.reset()
        if self.smoothed_field:
            self.smoothed_field.request_seed()
        zero_rule = np.zeros((10, 8), dtype=np.float32)
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

    def _handle_sweep_preview_restore(self, ui_state):
        """Handle sweep preview restore: ANY click re-enables sweeps. Returns True if restored."""
        if ui_state.sim.sweep_preview_pending_restore:
            if ui_state.any_left_click_this_frame or ui_state.any_right_click_this_frame:
                ui_state.sim.parameter_sweeps_enabled = True
                ui_state.sim.sweep_preview_pending_restore = False
                return True
        return False

    def _handle_mouse_clicks(self, ui_state, tiling_mode, canvas_aspect_ratio):
        """Handle left/right mouse click behavior based on mode."""
        if ui_state.left_click_this_frame:
            if ui_state.sim.parameter_sweeps_enabled:
                self._handle_sweep_click(ui_state, tiling_mode, canvas_aspect_ratio)
            elif ui_state.preferences.mouse_mode == "Select Particle":
                self._handle_entity_pick(ui_state, tiling_mode, canvas_aspect_ratio)

        elif ui_state.right_click_this_frame:
            if ui_state.sim.parameter_sweeps_enabled:
                if self.sim.has_active_xy_sweep() or self.sim.has_active_cohort_sweep():
                    ui_state.sim.parameter_sweeps_enabled = False
                    ui_state.sim.sweep_preview_pending_restore = True
            elif ui_state.preferences.mouse_mode == "Select Particle":
                if self.rule_manager.length() > 1:
                    prev_rule, prev_seed = self.rule_manager.pop_rule()
                    if prev_seed is not None:
                        ui_state.sim.rule_seed = prev_seed
                    self.sim.apply_rule(prev_rule)

    def _handle_sweep_click(self, ui_state, tiling_mode, canvas_aspect_ratio):
        """Handle left click when parameter sweeps are enabled."""
        if not (self.sim.has_active_xy_sweep() or self.sim.has_active_cohort_sweep()):
            return

        tex_coords = self.camera.screen_to_tex(
            ui_state.mouse_pos, self.sim.view_tex.size
        )
        if tiling_mode:
            tex_coords = (
                np.fmod(tex_coords[0] + 10.0, 1.0),
                np.fmod(tex_coords[1] + 10.0, 1.0)
            )
        world_pos = (tex_coords[0] * 2 - 1, tex_coords[1] * 2 - 1)

        if self.sim.has_active_cohort_sweep():
            entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(
                tex_coords, canvas_aspect_ratio,
                num_cohorts=ui_state.sim.num_cohorts, active_count=self.sim.entity_count)
            self.sim.update_sliders_from_particle(world_pos, entity_cohort)
        else:
            self.sim.update_sliders_from_position(world_pos)

    def _handle_entity_pick(self, ui_state, tiling_mode, canvas_aspect_ratio):
        """Handle entity selection via left click in Select Particle mode."""
        tex_coords = self.camera.screen_to_tex(
            ui_state.mouse_pos, self.sim.view_tex.size
        )
        if tiling_mode:
            tex_coords = (
                np.fmod(tex_coords[0] + 10.0, 1.0),
                np.fmod(tex_coords[1] + 10.0, 1.0)
            )
        entity_id, entity_pos, entity_cohort = self.entity_picker.find_nearest_entity(
            tex_coords, canvas_aspect_ratio,
            num_cohorts=ui_state.sim.num_cohorts, active_count=self.sim.entity_count)

        if entity_id >= 0 and entity_id < self.sim.entity_count:
            print(f"Entity {entity_id} at pos {entity_pos}, cohort {entity_cohort} - requesting rule buffer update")
            self.sim.request_rule_buffer_update(entity_id)
            self._pending_entity_selection = (entity_id, entity_pos, entity_cohort)
        else:
            print(f"Warning: entity_id {entity_id} out of bounds (max: {self.sim.entity_count - 1})")

    def _handle_config_commands(self, ui_state):
        """Handle config save/load/delete commands."""
        fh = self.field_handler

        # Config save (Ctrl+C)
        if ui_state.request_save_config:
            current_rule = self.rule_manager.get_current_rule()

            field_snapshot, field_strengths = (
                fh.snapshot_with_strengths(ui_state) if fh else (None, None))

            config = self.config_saver.create_config(
                ui_state.sim, current_rule, field_strengths=field_strengths)
            config_string = self.config_saver.encode_clipboard(config)
            self.ui.set_clipboard(config_string)
            self.ui.add_to_config_clipboard(
                config, self.ui.currently_open_project, field_snapshot=field_snapshot)
            if fh:
                fh.enforce_snapshot_cap(self.ui.config_clipboard)
            print(f"Config copied to clipboard ({len(config_string)} chars)")

        # Config load (Ctrl+V)
        if ui_state.request_load_config:
            config_string = ui_state.clipboard_text
            if config_string:
                config = self.config_saver.decode_clipboard(config_string)
                if config is not None:
                    rule = self._apply_config_with_locks(config, ui_state)
                    self._push_and_apply_rule(rule, ui_state)
                    if fh:
                        fh.apply_last_copied(ui_state)
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
                    if fh:
                        fh.delete_field_png(filepath)
                    print(f"Config deleted: {filepath}")

    def _handle_file_save(self, ui_state):
        """Handle file save from menu, including field texture PNG."""
        filename = ui_state.save_filename
        if not filename:
            return

        fh = self.field_handler
        current_rule = self.rule_manager.get_current_rule()

        field_data, field_strengths = (
            fh.snapshot_with_strengths(ui_state) if fh else (None, None))

        config = self.config_saver.create_config(
            ui_state.sim, current_rule, field_strengths=field_strengths)
        filepath = self.user_configs_dir / f"{filename}.json"
        self.config_saver.save_to_file(config, filepath)

        if fh:
            fh.save_field_png(field_data, filename, self.user_configs_dir)
            fh.invalidate_cache(filepath)

        print(f"Config saved to {filepath}")
        self.ui.update_physics_defaults(filename)

    def _handle_file_load(self, ui_state):
        """Handle file load from menu, including multi-load and preview modes."""
        filename = ui_state.load_filename
        category = ui_state.load_category
        if not filename:
            return

        fh = self.field_handler

        # Multi-load mode (ignores fields entirely)
        if ui_state.multi_load.multi_load_enabled:
            filepath = self.ui._get_config_path(filename, category)
            config = self.config_saver.load_from_file(filepath)
            if config is not None:
                success = self.multi_load_service.add_config(config, filename)
                if success:
                    print(f"Config added to multi-load: {filename}")
                else:
                    print(f"Failed to add config: multi-load list is full "
                          f"({self.multi_load_service.get_config_count()}/64)")
            else:
                print(f"Failed to load config from {filepath}")
            return

        # Normal mode
        if self.preview_rule_active:
            # Preview already applied config, rule, and field texture - just finalize
            self.preview_rule_active = False
            self._preview_rule_was_pushed = False
            if fh:
                fh.discard_preview_cache()
            if ui_state.load_watercolor_override is not None:
                ui_state.sim.watercolor_mode = ui_state.load_watercolor_override
            print(f"Config loaded (from preview): {filename}")
            self.ui.update_physics_defaults(filename)
        else:
            # No preview active - load fresh from file
            filepath = self.ui._get_config_path(filename, ui_state.load_category)
            config = self.config_saver.load_from_file(filepath)
            if config is not None:
                rule = self._apply_config_with_locks(
                    config, ui_state,
                    watercolor_override=ui_state.load_watercolor_override
                )
                self._push_and_apply_rule(rule, ui_state)
                if fh:
                    fh.apply_for_config(config, filepath, ui_state)
                print(f"Config loaded from {filepath}")
                self.ui.update_physics_defaults(filename)
            else:
                print(f"Failed to load config from {filepath}")

    def _handle_preview_commands(self, ui_state):
        """Handle config preview (hover in Load submenu) and clear preview."""
        fh = self.field_handler

        # Clear preview must happen before new preview
        if ui_state.request_clear_preview:
            if self.preview_rule_active:
                if self._preview_rule_was_pushed:
                    prev_rule, prev_seed = self.rule_manager.pop_rule()
                    if prev_seed is not None:
                        ui_state.sim.rule_seed = prev_seed
                    self.sim.apply_rule(prev_rule)
                self.preview_rule_active = False
                self._preview_rule_was_pushed = False

                if fh:
                    fh.restore_from_preview(ui_state)

        # New preview
        if ui_state.request_preview_config:
            filename = ui_state.preview_filename
            category = ui_state.preview_category
            if filename:
                filepath = self.ui._get_config_path(filename, category)
                config = self.config_saver.load_from_file(filepath)
                if config and config.rule is not None:
                    # Cache field texture on first preview entry
                    if not self.preview_rule_active and fh:
                        fh.cache_for_preview(ui_state)

                    pls = self.param_lock_service
                    if not (pls and pls.should_block_rule_push()):
                        if not (pls and pls.is_locked('rule_seed')):
                            ui_state.sim.rule_seed = config.rule_seed
                        self.rule_manager.push_rule(config.rule, ui_state.sim.rule_seed)
                        self.sim.apply_rule(config.rule)
                        self._preview_rule_was_pushed = True
                    self.preview_rule_active = True

                    if fh:
                        fh.apply_for_config(config, filepath, ui_state)

    def _handle_clipboard_commands(self, ui_state):
        """Handle config clipboard preview, load, delete, and import-to-multiload."""
        fh = self.field_handler

        # Clear clipboard preview (must happen before new preview)
        if ui_state.request_clear_clipboard_preview:
            if self.clipboard_preview_active:
                if self._clipboard_rule_was_pushed:
                    self.rule_manager.pop_rule()
                # Restore the full cached config (not just the rule)
                if self._clipboard_cached_config is not None:
                    rule = self._apply_config_with_locks(
                        self._clipboard_cached_config, ui_state)
                    self.sim.apply_rule(rule)
                    self._clipboard_cached_config = None

                if fh:
                    fh.restore_from_clipboard_preview(ui_state)

                self.clipboard_preview_active = False
                self._clipboard_rule_was_pushed = False

        # New clipboard preview
        if ui_state.request_preview_clipboard_config:
            idx = ui_state.clipboard_config_index
            if 0 <= idx < len(self.ui.config_clipboard):
                # Cache current full config before applying preview
                if not self.clipboard_preview_active:
                    current_rule = self.rule_manager.get_current_rule()
                    self._clipboard_cached_config = self.config_saver.create_config(
                        ui_state.sim, current_rule)
                    if fh:
                        fh.cache_for_clipboard_preview(ui_state)

                config, _label, field_snapshot = self.ui.config_clipboard[idx]
                rule = self._apply_config_with_locks(config, ui_state)
                pls = self.param_lock_service
                if not (pls and pls.should_block_rule_push()):
                    self.rule_manager.push_rule(rule, ui_state.sim.rule_seed)
                    self.sim.apply_rule(rule)
                    self._clipboard_rule_was_pushed = True
                self.clipboard_preview_active = True

                if fh:
                    fh.apply_snapshot(field_snapshot, config, ui_state)

        # Load clipboard config (click)
        if ui_state.request_load_clipboard_config:
            self._load_clipboard_config(ui_state)

        # Delete clipboard entry
        if ui_state.request_delete_clipboard_config:
            self._delete_clipboard_config(ui_state)

        # Import clipboard to multi-load
        if ui_state.request_import_clipboard_to_multiload:
            self._import_clipboard_to_multiload()

    def _load_clipboard_config(self, ui_state):
        """Load a config from the clipboard (apply it permanently)."""
        fh = self.field_handler

        # Clear preview first (discard cached config since we're committing)
        if self.clipboard_preview_active:
            if self._clipboard_rule_was_pushed:
                self.rule_manager.pop_rule()
            self.clipboard_preview_active = False
            self._clipboard_rule_was_pushed = False
            self._clipboard_cached_config = None
            if fh:
                fh.discard_clipboard_preview_cache()

        idx = ui_state.clipboard_config_index
        if 0 <= idx < len(self.ui.config_clipboard):
            config, label, field_snapshot = self.ui.config_clipboard[idx]
            rule = self._apply_config_with_locks(config, ui_state)
            self._push_and_apply_rule(rule, ui_state)

            if fh:
                fh.apply_snapshot(field_snapshot, config, ui_state)

            # Extract original filename from label (everything before the *)
            original_filename = label.rsplit("*", 1)[0]
            self.ui.update_physics_defaults(original_filename)
            print(f"Config loaded from clipboard: {label}")

    def _delete_clipboard_config(self, ui_state):
        """Delete an entry from the config clipboard."""
        fh = self.field_handler

        # Clear preview first, restore cached config
        if self.clipboard_preview_active:
            if self._clipboard_rule_was_pushed:
                self.rule_manager.pop_rule()
            if self._clipboard_cached_config is not None:
                rule = self._apply_config_with_locks(
                    self._clipboard_cached_config, ui_state)
                self.sim.apply_rule(rule)
                self._clipboard_cached_config = None

            if fh:
                fh.restore_from_clipboard_preview(ui_state)

            self.clipboard_preview_active = False
            self._clipboard_rule_was_pushed = False

        idx = ui_state.clipboard_config_index
        if 0 <= idx < len(self.ui.config_clipboard):
            self.ui.config_clipboard.pop(idx)

    def _import_clipboard_to_multiload(self):
        """Replace multi-load configs with contents of config clipboard."""
        # Clear existing multi-load configs
        while self.multi_load_service.get_config_count() > 0:
            self.multi_load_service.remove_config(0)

        # Add each clipboard entry
        for config, label, _field in self.ui.config_clipboard:
            self.multi_load_service.add_config(config, label)

        print(f"Imported {len(self.ui.config_clipboard)} configs from clipboard to multi-load")
