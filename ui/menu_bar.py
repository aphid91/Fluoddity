"""Main menu bar: File, Reset, Help, Extras menus with auto-close logic."""
from imgui_bundle import imgui


class MenuBarMixin:
    """Mixin for main menu bar. Combined into UI via multiple inheritance."""

    def _apply_config_locked(self, config, watercolor_override=None):
        """Apply config with parameter lock snapshot/restore."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(self.state.sim, self.state.preferences) if pls else {}
        self.config_saver.apply_config(config, self.state.sim,
                                       watercolor_override=watercolor_override)
        if pls:
            pls.restore_locked(self.state.sim, self.state.preferences, snapshot)

    def _load_from_string_locked(self, config_string, watercolor_override=None):
        """Load config from string with parameter lock snapshot/restore."""
        pls = self.param_lock_service
        snapshot = pls.snapshot_locked(self.state.sim, self.state.preferences) if pls else {}
        self.config_saver.load_from_string(config_string, self.state.sim,
                                           watercolor_override=watercolor_override)
        if pls:
            pls.restore_locked(self.state.sim, self.state.preferences, snapshot)

    def render_main_menu_bar(self):
        """Render the main application menu bar at the top of the window."""
        load_submenu_open = False
        any_menu_open_this_frame = False

        if imgui.begin_main_menu_bar():
            # Track all open menu rectangles separately (not combined into one giant box)
            # We'll calculate distance as the minimum distance to any of these rectangles
            menu_rectangles = []

            # Start with the menu bar itself
            menu_bar_min = imgui.get_window_pos()
            menu_bar_size = imgui.get_window_size()
            menu_rectangles.append((menu_bar_min.x, menu_bar_min.y,
                                   menu_bar_min.x + menu_bar_size.x,
                                   menu_bar_min.y + menu_bar_size.y))
            if imgui.begin_menu("File", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                file_menu_min = imgui.get_window_pos()
                file_menu_size = imgui.get_window_size()
                menu_rectangles.append((file_menu_min.x, file_menu_min.y,
                                       file_menu_min.x + file_menu_size.x,
                                       file_menu_min.y + file_menu_size.y))

                if imgui.menu_item("New", "", False)[0]:
                    self._load_filename = "_Default"
                    self._load_category = "Core"
                    self._request_load_file = True
                    self._load_watercolor_override = None
                self._delayed_tooltip("Start a fresh config. Loads from _Default")

                imgui.separator()

                if imgui.menu_item("Save...", "", False)[0]:
                    self.save_popup_open = True
                    # Default to last loaded filename
                    self.save_filename_buffer = self.currently_open_project
                self._delayed_tooltip("Save the current physics settings including particle rules.")

                # Load submenu with preview - locks to current watercolor mode
                # Right-click toggles watercolor mode
                if imgui.begin_menu("Load", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    load_submenu_open = True
                    # Add this submenu's bounding box to the list
                    load_menu_min = imgui.get_window_pos()
                    load_menu_size = imgui.get_window_size()
                    menu_rectangles.append((load_menu_min.x, load_menu_min.y,
                                           load_menu_min.x + load_menu_size.x,
                                           load_menu_min.y + load_menu_size.y))

                    # First frame submenu opens: cache current state and scan config files
                    if not self.load_submenu_was_open:
                        self._cache_all_configs()
                        # Cache current config as JSON string for restoration
                        self.cached_config = self.config_saver.save_to_string(
                            self.state.sim, self._display_info.get('current_rule'))
                        self.preview_rule_pushed = False
                        self.currently_previewing = None
                        self.currently_previewing_category = None
                        # Lock to current watercolor mode when menu opens
                        self.load_menu_watercolor_mode = self.state.sim.watercolor_mode
                        # Cache field strengths for restoration
                        self._cached_field_strengths = (
                            self.state.preferences.force_field_strength,
                            self.state.preferences.strafe_field_strength,
                        )

                    # Use locked watercolor mode
                    current_menu_watercolor = self.load_menu_watercolor_mode

                    # Header showing right-click hint (compact two-line format)
                    mode_text = "Watercolor ON" if current_menu_watercolor else "Watercolor OFF"
                    imgui.text_disabled("Right-click toggles:")
                    imgui.text_disabled(f"({mode_text})")
                    imgui.separator()

                    # Check for right-click anywhere in the menu to toggle watercolor
                    if imgui.is_window_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.load_menu_watercolor_mode = not self.load_menu_watercolor_mode
                        current_menu_watercolor = self.load_menu_watercolor_mode
                        # Update any current preview with new watercolor mode
                        if self.currently_previewing and self.currently_previewing_category:
                            cache_key = f"{self.currently_previewing_category}/{self.currently_previewing}"
                            if cache_key in self.cached_configs:
                                config = self.cached_configs[cache_key]
                                self._apply_config_locked(config,
                                                         watercolor_override=current_menu_watercolor)
                                # Re-apply field strengths from config (respect locks)
                                if config.force_field_strength is not None:
                                    pls = self.param_lock_service
                                    if not (pls and pls.is_locked('force_field_strength')):
                                        self.state.preferences.force_field_strength = config.force_field_strength
                                    if not (pls and pls.is_locked('strafe_field_strength')):
                                        self.state.preferences.strafe_field_strength = config.strafe_field_strength
                        elif self.cached_config:
                            # Restore from cache with watercolor override
                            self._load_from_string_locked(
                                self.cached_config,
                                watercolor_override=current_menu_watercolor)
                            # Restore cached field strengths
                            if self._cached_field_strengths is not None:
                                self.state.preferences.force_field_strength = self._cached_field_strengths[0]
                                self.state.preferences.strafe_field_strength = self._cached_field_strengths[1]

                    # Lock watercolor mode to menu's mode
                    self.state.sim.watercolor_mode = current_menu_watercolor

                    hovered_this_frame = self._render_load_submenu_content(current_menu_watercolor)

                    # Handle preview on hover (works in both normal and multi-load modes)
                    # hovered_this_frame is now a tuple (filename, category) or None
                    hovered_filename = hovered_this_frame[0] if hovered_this_frame else None
                    hovered_category = hovered_this_frame[1] if hovered_this_frame else None
                    current_preview = (self.currently_previewing, self.currently_previewing_category)

                    if hovered_this_frame != current_preview:
                        # First, clear any existing preview
                        if self.currently_previewing:
                            self._request_clear_preview = True

                        if hovered_filename and hovered_category:
                            cache_key = f"{hovered_category}/{hovered_filename}"
                            if cache_key in self.cached_configs:
                                # Apply preview config with watercolor override
                                config = self.cached_configs[cache_key]
                                self._apply_config_locked(config,
                                                         watercolor_override=current_menu_watercolor)
                                # Apply field strengths from config if present (respect locks)
                                if config.force_field_strength is not None:
                                    pls = self.param_lock_service
                                    if not (pls and pls.is_locked('force_field_strength')):
                                        self.state.preferences.force_field_strength = config.force_field_strength
                                    if not (pls and pls.is_locked('strafe_field_strength')):
                                        self.state.preferences.strafe_field_strength = config.strafe_field_strength
                                self._request_preview_config = True
                                self._preview_filename = hovered_filename
                                self._preview_category = hovered_category
                                self.currently_previewing = hovered_filename
                                self.currently_previewing_category = hovered_category
                        elif hovered_this_frame is None and self.cached_config:
                            # Revert to cached state with watercolor override
                            self._load_from_string_locked(
                                self.cached_config,
                                watercolor_override=current_menu_watercolor)
                            # Restore cached field strengths
                            if self._cached_field_strengths is not None:
                                self.state.preferences.force_field_strength = self._cached_field_strengths[0]
                                self.state.preferences.strafe_field_strength = self._cached_field_strengths[1]
                            self.currently_previewing = None
                            self.currently_previewing_category = None

                    imgui.end_menu()

                imgui.separator()

                # Preferences toggle
                if imgui.menu_item("Preferences", "", self.state.preferences.show_preferences_window)[0]:
                    self.state.preferences.show_preferences_window = not self.state.preferences.show_preferences_window

                imgui.end_menu()

            # Window toggle button (shows/hides Physics Settings, Preferences, Drawing Controls, Config Clipboard, Screen Recording)
            if imgui.menu_item("Show/Hide Windows (X)", "", self.show_sidebar)[0]:
                self.show_sidebar = not self.show_sidebar

            # Reset menu
            if imgui.begin_menu("Reset...", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                reset_menu_min = imgui.get_window_pos()
                reset_menu_size = imgui.get_window_size()
                menu_rectangles.append((reset_menu_min.x, reset_menu_min.y,
                                       reset_menu_min.x + reset_menu_size.x,
                                       reset_menu_min.y + reset_menu_size.y))

                # Revert to current project (reload the file)
                revert_label = f"Revert to '{self.currently_open_project}'"

                if imgui.menu_item(revert_label, "", False)[0]:
                    # Trigger file load equivalent to File->Load
                    self._load_filename = self.currently_open_project
                    self._request_load_file = True
                    self._load_watercolor_override = self.state.sim.watercolor_mode  # Preserve current watercolor mode
                self._delayed_tooltip(f"Equivalent to File -> Load {self.currently_open_project}")

                # Reset all slider ranges
                if imgui.menu_item("Reset all slider ranges to defaults", "", False)[0]:
                    # Clear all custom slider ranges, reverting to defaults
                    self.state.sim.slider_ranges.clear()

                # Reset all parameter sweeps
                if imgui.menu_item("Reset all parameter sweeps", "", False)[0]:
                    # Turn off all parameter sweeps
                    for param in list(self.state.sim.x_sweeps.keys()):
                        self.state.sim.x_sweeps[param] = 0.0
                        self.state.sim.y_sweeps[param] = 0.0
                        self.state.sim.cohort_sweeps[param] = 0.0
                self._delayed_tooltip("Set all parameter sweeps to 'off'.")

                # Reset all UI settings
                if imgui.menu_item("Reset all UI settings", "", False)[0]:
                    # Reset preferences to defaults (equivalent to deleting preferences.config)
                    from state.preferences_state import PreferencesState
                    self.state.preferences = PreferencesState()
                self._delayed_tooltip("Restore all preferences and ui state to factory settings. \nEquivalent to deleting preferences.config, or running this\nprogram for the first time. Physics config saves are not affected.")

                # Reset camera
                if imgui.menu_item("Reset camera", "", False)[0]:
                    self._request_camera_reset = True
                self._delayed_tooltip("Return camera to default position and zoom level.")

                # Reset canvas and fields
                if imgui.menu_item("Reset Canvas and Fields", "", False)[0]:
                    self._request_clear_canvas_and_fields = True
                self._delayed_tooltip("Clear canvas, brush, and all field textures to zero.")

                imgui.end_menu()

            # Parameter Locks menu (only visible when enabled)
            pls = self.param_lock_service
            if pls and self.state.preferences.parameter_locks_enabled:
                if imgui.begin_menu("Locks", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    locks_menu_min = imgui.get_window_pos()
                    locks_menu_size = imgui.get_window_size()
                    menu_rectangles.append((locks_menu_min.x, locks_menu_min.y,
                                           locks_menu_min.x + locks_menu_size.x,
                                           locks_menu_min.y + locks_menu_size.y))

                    # Lock/Unlock everything (dynamic label)
                    if pls.any_locked:
                        if imgui.menu_item("Unlock Everything", "", False)[0]:
                            pls.unlock_all()
                    else:
                        if imgui.menu_item("Lock Everything", "", False)[0]:
                            pls.lock_all()

                    imgui.separator()

                    changed, pls.lock_rule = imgui.checkbox("Lock Rule", pls.lock_rule)
                    if changed:
                        pls._locks['rule_seed'] = pls.lock_rule
                    self._delayed_tooltip("Prevent the target rule and mutation seed\nfrom being changed by config loads/pastes.\nMutation seed can also be locked independently via Alt-click.")

                    _, pls.lock_force_field = imgui.checkbox(
                        "Lock Force Field", pls.lock_force_field)
                    self._delayed_tooltip("Prevent the force components of the field\ntexture from being changed by loads/pastes.")

                    _, pls.lock_strafe_field = imgui.checkbox(
                        "Lock Strafe Field", pls.lock_strafe_field)
                    self._delayed_tooltip("Prevent the strafe components of the field\ntexture from being changed by loads/pastes.")

                    imgui.end_menu()

            # Help menu
            if imgui.begin_menu("Help", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                help_menu_min = imgui.get_window_pos()
                help_menu_size = imgui.get_window_size()
                menu_rectangles.append((help_menu_min.x, help_menu_min.y,
                                       help_menu_min.x + help_menu_size.x,
                                       help_menu_min.y + help_menu_size.y))

                if imgui.menu_item("Guide", "", self.state.preferences.show_tutorial_window)[0]:
                    self.state.preferences.show_tutorial_window = not self.state.preferences.show_tutorial_window
                if imgui.menu_item("Controls", "", self.state.preferences.show_controls_window)[0]:
                    self.state.preferences.show_controls_window = not self.state.preferences.show_controls_window
                if imgui.menu_item("Performance", "", self.state.preferences.show_performance_window)[0]:
                    self.state.preferences.show_performance_window = not self.state.preferences.show_performance_window
                if imgui.menu_item("Parameter Sweeps", "", self.state.preferences.show_parameter_sweeps_window)[0]:
                    self.state.preferences.show_parameter_sweeps_window = not self.state.preferences.show_parameter_sweeps_window
                imgui.end_menu()

            # Extras menu
            if imgui.begin_menu("Extras", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                extras_menu_min = imgui.get_window_pos()
                extras_menu_size = imgui.get_window_size()
                menu_rectangles.append((extras_menu_min.x, extras_menu_min.y,
                                       extras_menu_min.x + extras_menu_size.x,
                                       extras_menu_min.y + extras_menu_size.y))

                # Config Clipboard window
                _, self.show_history_window = imgui.checkbox(
                    "Config Clipboard",
                    self.show_history_window
                )
                self._delayed_tooltip("Set restorable checkpoints with Ctrl-C")

                # Screen Recording Controls
                _, self.show_video_recording_window = imgui.checkbox(
                    "Screen Recording Controls",
                    self.show_video_recording_window
                )

                # Load Field submenu
                if imgui.begin_menu("Load Field"):
                    if imgui.menu_item("Load Force Field...", "", False)[0]:
                        self._open_field_loader("force")
                    self._delayed_tooltip("Load a PNG/JPEG image as a force field.\nR=magnitude, G=angle (polar coordinates).")
                    if imgui.menu_item("Load Strafe Field...", "", False)[0]:
                        self._open_field_loader("strafe")
                    self._delayed_tooltip("Load a PNG/JPEG image as a strafe field.\nR=magnitude, G=angle (polar coordinates).")
                    imgui.end_menu()

                # Strong Determinism toggle
                _, self.state.preferences.strong_determinism = imgui.checkbox(
                    "Strong Determinism",
                    self.state.preferences.strong_determinism
                )
                self._delayed_tooltip("Enables double buffering for the canvas. When checked,\nevents will unfold exactly the same way after every\nsimulation reset. Comes with a small ~3% performance penalty.")

                # Advanced Drawing toggle
                _, self.state.preferences.advanced_drawing_enabled = imgui.checkbox(
                    "Advanced Drawing - EXPERIMENTAL",
                    self.state.preferences.advanced_drawing_enabled
                )
                self._delayed_tooltip(
                    "Open the Drawing Controls window for advanced\n"
                    "brush modes, force fields, and strafe fields."
                )

                # Streamline toggle
                _, self.state.streamline.enabled = imgui.checkbox(
                    "Streamline",
                    self.state.streamline.enabled
                )
                self._delayed_tooltip(
                    "Trace the path of a free particle released at the cursor\n"
                    "and carried by the canvas force field."
                )

                # Multi Load toggle
                _, self.state.multi_load.multi_load_enabled = imgui.checkbox(
                    "Multi Load - EXPERIMENTAL",
                    self.state.multi_load.multi_load_enabled
                )
                self._delayed_tooltip("Load multiple files at once, so that particles\nfrom different saves can interact.")

                # Parameter Locks checkbox (greyed out in multiload mode)
                multiload_active = self.state.multi_load.multi_load_enabled
                if multiload_active:
                    imgui.begin_disabled()
                changed, new_val = imgui.checkbox(
                    "Parameter Locks - EXPERIMENTAL",
                    self.state.preferences.parameter_locks_enabled
                )
                if changed:
                    self.state.preferences.parameter_locks_enabled = new_val
                    if self.param_lock_service:
                        if new_val:
                            self.param_lock_service.enabled = True
                        else:
                            self.param_lock_service.reset()
                self._delayed_tooltip(
                    "Alt-Click on a parameter to freeze it and its value\n"
                    "won't change when loading new configs.")
                if multiload_active:
                    imgui.end_disabled()

                imgui.end_menu()

            # After all menus: check mouse distance from all menu rectangles
            # Find the minimum distance to any rectangle
            if self.main_menu_bar_has_open_menu and not self.save_popup_open:
                mouse_pos = imgui.get_mouse_pos()

                # Calculate minimum distance to any menu rectangle
                min_distance = float('inf')
                for min_x, min_y, max_x, max_y in menu_rectangles:
                    dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                    dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # If mouse is too far away from all rectangles, signal to close menus
                if min_distance > self.state.preferences.menu_close_threshold:
                    self.force_close_main_menus = True

            imgui.end_main_menu_bar()

        # Update menu tracking state
        self.main_menu_bar_has_open_menu = any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_main_menus and not any_menu_open_this_frame:
            self.force_close_main_menus = False

        # Handle submenu close without selection
        if self.load_submenu_was_open and not load_submenu_open:
            # Submenu just closed - restore cached state (no watercolor override)
            if self.cached_config:
                self._load_from_string_locked(self.cached_config)
            if self.currently_previewing:
                self._request_clear_preview = True
            # Restore cached field strengths
            if hasattr(self, '_cached_field_strengths') and self._cached_field_strengths is not None:
                self.state.preferences.force_field_strength = self._cached_field_strengths[0]
                self.state.preferences.strafe_field_strength = self._cached_field_strengths[1]
                self._cached_field_strengths = None
            self.cached_config = None
            self.currently_previewing = None
            self.currently_previewing_category = None
            self.cached_configs = {}
            self.preview_rule_pushed = False
            self.load_menu_watercolor_mode = None  # Clear the watercolor lock

        self.load_submenu_was_open = load_submenu_open
