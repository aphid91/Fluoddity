"""Main menu bar: File, Reset, Help, Extras menus with auto-close logic."""
from imgui_bundle import imgui


class MenuBarMixin:
    """Mixin for main menu bar. Combined into UI via multiple inheritance."""

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
                self._delayed_tooltip("Start a fresh config. Loads from _Default")

                imgui.separator()

                if imgui.menu_item("Save...", "", False)[0]:
                    self.save_popup_open = True
                    # Default to last loaded filename
                    self.save_filename_buffer = self.currently_open_project
                self._delayed_tooltip("Save the current physics settings including particle rules.")

                # Load submenu with preview
                if imgui.begin_menu("Load", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    load_submenu_open = True
                    # Add this submenu's bounding box to the list
                    load_menu_min = imgui.get_window_pos()
                    load_menu_size = imgui.get_window_size()
                    menu_rectangles.append((load_menu_min.x, load_menu_min.y,
                                           load_menu_min.x + load_menu_size.x,
                                           load_menu_min.y + load_menu_size.y))

                    # First frame submenu opens: scan config files and reset hover
                    # state. All config apply/restore is delegated to CommandHandler
                    # via one-shot flags — the UI only tracks what is being previewed.
                    if not self.load_submenu_was_open:
                        self._cache_all_configs()
                        self.currently_previewing = None
                        self.currently_previewing_category = None

                    hovered_this_frame = self._render_load_submenu_content()

                    # Handle preview on hover — flags only; CommandHandler applies.
                    # hovered_this_frame is a (filename, category) tuple or None.
                    hovered_filename = hovered_this_frame[0] if hovered_this_frame else None
                    hovered_category = hovered_this_frame[1] if hovered_this_frame else None
                    current_preview = (self.currently_previewing, self.currently_previewing_category)

                    if hovered_this_frame != current_preview:
                        # First, clear any existing preview (restore original)
                        if self.currently_previewing:
                            self._request_clear_preview = True

                        if hovered_filename and hovered_category:
                            # Request the preview load; CommandHandler reads the
                            # config from disk, caches the original, and applies.
                            self._request_preview_config = True
                            self._preview_filename = hovered_filename
                            self._preview_category = hovered_category
                            self.currently_previewing = hovered_filename
                            self.currently_previewing_category = hovered_category
                        else:
                            # Hover left all items — restore requested above.
                            self.currently_previewing = None
                            self.currently_previewing_category = None

                    imgui.end_menu()

                imgui.end_menu()

            # Editor menu — non-physics editor state (preferences + render
            # settings + imgui layout) save/load, plus the Preferences toggle.
            if imgui.begin_menu("Editor", not self.force_close_main_menus):
                any_menu_open_this_frame = True
                editor_menu_min = imgui.get_window_pos()
                editor_menu_size = imgui.get_window_size()
                menu_rectangles.append((editor_menu_min.x, editor_menu_min.y,
                                       editor_menu_min.x + editor_menu_size.x,
                                       editor_menu_min.y + editor_menu_size.y))

                # Preferences toggle (moved here from File).
                if imgui.menu_item("Preferences", "", self.state.preferences.ui_windows.show_preferences_window)[0]:
                    self.state.preferences.ui_windows.show_preferences_window = not self.state.preferences.ui_windows.show_preferences_window

                # Render settings toggle (moved here from Extras).
                if imgui.menu_item("Render settings", "", self.state.preferences.ui_windows.show_render_settings_window)[0]:
                    self.state.preferences.ui_windows.show_render_settings_window = not self.state.preferences.ui_windows.show_render_settings_window
                self._delayed_tooltip("Per-renderer controls: RT mode, capture, camera,\nmedium/geometry, lighting, sky, and post-process.\nShows the active renderer's settings (Preferences -> Renderer).")

                imgui.separator()

                # Save Editor Settings... (opens the editor-save popup)
                if imgui.menu_item("Save Editor Settings...", "", False)[0]:
                    self.editor_save_popup_open = True
                    self._save_editor_name_buffer = self._save_editor_name or "editor"
                self._delayed_tooltip(
                    "Save all non-physics settings (preferences, render settings,\n"
                    "window visibility, and window/docking layout).")

                # Load Editor Settings submenu (no hover preview)
                if imgui.begin_menu("Load Editor Settings", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    le_min = imgui.get_window_pos()
                    le_size = imgui.get_window_size()
                    menu_rectangles.append((le_min.x, le_min.y,
                                           le_min.x + le_size.x, le_min.y + le_size.y))
                    if not self._editor_load_submenu_was_open:
                        self._refresh_editor_save_files()
                    if not self._editor_save_files:
                        imgui.text_disabled("(no saves)")
                    else:
                        labels = [p.name[:-len('.editor.json')] for p in self._editor_save_files]
                        max_w = max((imgui.calc_text_size(l).x for l in labels), default=0.0)
                        for path, label in zip(self._editor_save_files, labels):
                            clicked, _ = imgui.selectable(
                                f"{label}##editor_load", False,
                                imgui.SelectableFlags_.no_auto_close_popups,
                                imgui.ImVec2(max_w + 10, 0))
                            if clicked:
                                self._request_load_editor = True
                                self._load_editor_path = str(path)
                                imgui.close_current_popup()
                            imgui.same_line()
                            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
                            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                            if imgui.small_button(f"X##editor_del_{label}"):
                                self._editor_delete_path = path
                            imgui.pop_style_color(2)
                    imgui.end_menu()
                    self._editor_load_submenu_was_open = True
                else:
                    self._editor_load_submenu_was_open = False

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
                    # Route through the command handler: reset prefs to factory
                    # defaults (identity-preserved), then overlay the project's
                    # __Default_Editor save (prefs + docking) if present.
                    self._request_reset_ui_settings = True
                self._delayed_tooltip("Restore all preferences and ui state to factory settings. \nEquivalent to deleting preferences.config, or running this\nprogram for the first time. Physics config saves are not affected.")

                # Reset camera
                if imgui.menu_item("Reset camera", "", False)[0]:
                    self._request_camera_reset = True
                self._delayed_tooltip("Return camera to default position and zoom level.")

                # Reset canvas
                if imgui.menu_item("Reset Canvas", "", False)[0]:
                    self.state.request_clear_canvas = True
                self._delayed_tooltip("Clear the trail canvas to zero.")

                imgui.end_menu()

            # Parameter Locks menu (only visible when enabled)
            pls = self.param_lock_service
            if pls and self.state.preferences.parameter_locks.enabled:
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

                if imgui.menu_item("Guide", "", self.state.preferences.ui_windows.show_tutorial_window)[0]:
                    self.state.preferences.ui_windows.show_tutorial_window = not self.state.preferences.ui_windows.show_tutorial_window
                if imgui.menu_item("Controls", "", self.state.preferences.ui_windows.show_controls_window)[0]:
                    self.state.preferences.ui_windows.show_controls_window = not self.state.preferences.ui_windows.show_controls_window
                if imgui.menu_item("Performance", "", self.state.preferences.ui_windows.show_performance_window)[0]:
                    self.state.preferences.ui_windows.show_performance_window = not self.state.preferences.ui_windows.show_performance_window
                if imgui.menu_item("Parameter Sweeps", "", self.state.preferences.ui_windows.show_parameter_sweeps_window)[0]:
                    self.state.preferences.ui_windows.show_parameter_sweeps_window = not self.state.preferences.ui_windows.show_parameter_sweeps_window
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
                _, self.state.preferences.ui_windows.show_config_clipboard_window = imgui.checkbox(
                    "Config Clipboard",
                    self.state.preferences.ui_windows.show_config_clipboard_window
                )
                self._delayed_tooltip("Set restorable checkpoints with Ctrl-C")

                # Screen Recording Controls
                _, self.state.preferences.ui_windows.show_video_recording_window = imgui.checkbox(
                    "Screen Recording Controls",
                    self.state.preferences.ui_windows.show_video_recording_window
                )

                # Parameter Locks checkbox
                changed, new_val = imgui.checkbox(
                    "Parameter Locks - EXPERIMENTAL",
                    self.state.preferences.parameter_locks.enabled
                )
                if changed:
                    self.state.preferences.parameter_locks.enabled = new_val
                    if self.param_lock_service:
                        if new_val:
                            self.param_lock_service.enabled = True
                        else:
                            self.param_lock_service.reset()
                self._delayed_tooltip(
                    "Alt-Click on a parameter to freeze it and its value\n"
                    "won't change when loading new configs.")

                imgui.separator()

                # Dev submenu (Generics + Plotting)
                if imgui.begin_menu("Dev", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    dev_min = imgui.get_window_pos()
                    dev_size = imgui.get_window_size()
                    menu_rectangles.append((dev_min.x, dev_min.y,
                                           dev_min.x + dev_size.x,
                                           dev_min.y + dev_size.y))

                    # Generics window toggle
                    _, self.state.preferences.ui_windows.show_generics_window = imgui.checkbox(
                        "Generics",
                        self.state.preferences.ui_windows.show_generics_window
                    )
                    self._delayed_tooltip("DEV: Use generic03.xyzw and generic47.xyzw as variables in entity_update.glsl")

                    # Plotting window toggle
                    _, self.state.preferences.ui_windows.show_plotting_window = imgui.checkbox(
                        "Plotting",
                        self.state.preferences.ui_windows.show_plotting_window
                    )
                    self._delayed_tooltip("DEV: Use report() in entity_update.glsl to generate histograms")

                    imgui.end_menu()

                # Radio window toggle
                _, self.state.preferences.ui_windows.show_radio_window = imgui.checkbox(
                    "Radio",
                    self.state.preferences.ui_windows.show_radio_window
                )
                self._delayed_tooltip("Filter particle visibility by frequency band.\nOnly particles within the target frequency\n+/- bandwidth are visible.")

                # Scheduled Renders window toggle
                _, self.state.preferences.ui_windows.show_scheduled_renders_window = imgui.checkbox(
                    "Scheduled Renders",
                    self.state.preferences.ui_windows.show_scheduled_renders_window
                )
                self._delayed_tooltip("Queue multiple render specs for\nunattended batch video rendering.")

                imgui.separator()

                # Simulation state save/load (entity + canvas GPU buffers)
                if imgui.menu_item("Save Simulation State...", "", False)[0]:
                    self.simulation_save_popup_open = True
                    self._save_simulation_name_buffer = self._save_simulation_name or "simulation"
                self._delayed_tooltip(
                    "Dump the current entity buffer and 3D canvas\n"
                    "to disk (particle positions + trail densities).")

                if imgui.begin_menu("Load Simulation State", not self.force_close_main_menus):
                    any_menu_open_this_frame = True
                    ls_min = imgui.get_window_pos()
                    ls_size = imgui.get_window_size()
                    menu_rectangles.append((ls_min.x, ls_min.y,
                                           ls_min.x + ls_size.x, ls_min.y + ls_size.y))
                    if not self._simulation_load_submenu_was_open:
                        self._refresh_simulation_save_dirs()
                    if not self._simulation_save_dirs:
                        imgui.text_disabled("(no saves)")
                    else:
                        labels = [p.name[:-len('.fsim')] for p in self._simulation_save_dirs]
                        max_w = max((imgui.calc_text_size(l).x for l in labels), default=0.0)
                        for path, label in zip(self._simulation_save_dirs, labels):
                            clicked, _ = imgui.selectable(
                                f"{label}##sim_load", False,
                                imgui.SelectableFlags_.no_auto_close_popups,
                                imgui.ImVec2(max_w + 10, 0))
                            if clicked:
                                self._request_load_simulation = True
                                self._load_simulation_path = str(path)
                                imgui.close_current_popup()
                            imgui.same_line()
                            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
                            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                            if imgui.small_button(f"X##sim_del_{label}"):
                                self._simulation_delete_path = path
                            imgui.pop_style_color(2)
                    imgui.end_menu()
                    self._simulation_load_submenu_was_open = True
                else:
                    self._simulation_load_submenu_was_open = False

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
                if min_distance > self.state.preferences.ui_windows.menu_close_threshold:
                    self.force_close_main_menus = True

            imgui.end_main_menu_bar()

        # Update menu tracking state
        self.main_menu_bar_has_open_menu = any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_main_menus and not any_menu_open_this_frame:
            self.force_close_main_menus = False

        # Handle submenu close without selection
        if self.load_submenu_was_open and not load_submenu_open:
            # Submenu just closed - restore the original via CommandHandler.
            if self.currently_previewing:
                self._request_clear_preview = True
            self.currently_previewing = None
            self.currently_previewing_category = None
            self.cached_configs = {}

        self.load_submenu_was_open = load_submenu_open
