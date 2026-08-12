"""Physics Settings window: sliders, additional settings, appearance, notes (normal + multi-load modes)."""
from imgui_bundle import imgui
from .physics_params import PARAM_GROUPS


class PhysicsWindowMixin:
    """Mixin for physics settings window. Combined into UI via multiple inheritance."""

    def render_physics_settings_window(self):
        """Render the Physics Settings window with sliders."""
        # Apply bluish background when in sweep preview mode (waiting for click to restore sweeps)
        if self.state.sim.sweep_preview_pending_restore:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.15, 0.20, 0.35, 0.94))
            imgui.push_style_color(imgui.Col_.title_bg_active, imgui.ImVec4(0.20, 0.30, 0.50, 1.0))

        # No p_open parameter - window is uncloseable
        imgui.begin('Physics Settings', flags=imgui.WindowFlags_.menu_bar)

        # Track if any physics menu is open
        physics_any_menu_open_this_frame = False

        # Multi-load mode: different rendering
        if self.state.multi_load.multi_load_enabled:
            self._render_physics_multi_load_mode(physics_any_menu_open_this_frame)
            return

        # Normal mode: render sliders and settings
        physics_any_menu_open_this_frame = self._render_physics_normal_menu_bar()

        # Display currently open project and rule seed
        pls = self.param_lock_service
        imgui.text(f"Project: {self.currently_open_project}")
        imgui.same_line()
        # Convert rule_seed (0.0-1.0 float) to short hex format
        seed_hex = format(int(self.state.sim.rule_seed * 0xFFFF), '04x')
        seed_locked = pls and pls.is_locked('rule_seed')
        if seed_locked:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"[L]Mutation Seed: #{seed_hex}")
        else:
            imgui.text_colored(imgui.ImVec4(0.5, 0.8, 1.0, 1.0), f"Mutation Seed: #{seed_hex}")
        if pls:
            pls.handle_alt_click('rule_seed')
        imgui.separator()

        # === Basics Group (Trail sensors and rule mutation) ===
        imgui.set_next_item_open(self.state.preferences.physics_group_basics)
        basics_open = imgui.collapsing_header("Basics - Trail sensors and rule mutation")
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_basics = basics_open
        if basics_open:
            for pdef in PARAM_GROUPS['basics']:
                self.render_physics_slider(pdef)

        # === Forces Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_forces)
        forces_open = imgui.collapsing_header("Forces")
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_forces = forces_open
        if forces_open:
            for pdef in PARAM_GROUPS['forces']:
                self.render_physics_slider(pdef)

        # === Advanced Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_advanced)
        advanced_open = imgui.collapsing_header("Advanced")
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_advanced = advanced_open
        if advanced_open:
            for pdef in PARAM_GROUPS['advanced']:
                self.render_physics_slider(pdef)

        # === Additional Settings Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_additional)
        additional_open = imgui.collapsing_header("Additional Settings")
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_additional = additional_open
        if additional_open:
            # Boundary Conditions (with per-option tooltips)
            boundary_options = ["Bounce", "Reset", "Wrap"]
            boundary_tooltips = [
                "Particles bounce off the edges of the canvas",
                "Particles are reset to their initial conditions when leaving the canvas",
                "Particles wrap seamlessly to the other side of the canvas"
            ]
            bc_lock_colors = pls.push_locked_style('boundary_conditions') if pls else 0
            bc_label = pls.get_display_label('boundary_conditions', "Boundary Conditions") if pls else "Boundary Conditions"
            imgui.set_next_item_width(100)
            if imgui.begin_combo(bc_label, boundary_options[self.state.sim.boundary_conditions]):
                if pls and pls.begin_combo_alt_click('boundary_conditions'):
                    pass  # alt-click intercepted; combo closed
                else:
                    for i, option in enumerate(boundary_options):
                        is_selected = (self.state.sim.boundary_conditions == i)
                        if imgui.selectable(option, is_selected)[0]:
                            self.state.sim.boundary_conditions = i
                        self._delayed_tooltip(boundary_tooltips[i])
                        if is_selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()
            elif pls:
                pls.handle_alt_click('boundary_conditions')
            if pls:
                pls.pop_locked_style(bc_lock_colors)

            # Initial Conditions (with per-option tooltips)
            initial_options = ["Grid", "Random", "Ring"]
            initial_tooltips = [
                "Particles start in a grid, organized by cohort",
                "Particles are spread uniformly across the canvas",
                "Particles start distributed around a circle, organized by cohort"
            ]
            ic_lock_colors = pls.push_locked_style('initial_conditions') if pls else 0
            ic_label = pls.get_display_label('initial_conditions', "Initial Conditions") if pls else "Initial Conditions"
            imgui.set_next_item_width(100)
            if imgui.begin_combo(ic_label, initial_options[self.state.sim.initial_conditions]):
                if pls and pls.begin_combo_alt_click('initial_conditions'):
                    pass  # alt-click intercepted; combo closed
                else:
                    for i, option in enumerate(initial_options):
                        is_selected = (self.state.sim.initial_conditions == i)
                        if imgui.selectable(option, is_selected)[0]:
                            self.state.sim.initial_conditions = i
                        self._delayed_tooltip(initial_tooltips[i])
                        if is_selected:
                            imgui.set_item_default_focus()
                    imgui.end_combo()
            elif pls:
                pls.handle_alt_click('initial_conditions')
            if pls:
                pls.pop_locked_style(ic_lock_colors)

            # Number of Cohorts
            nc_lock_colors = pls.push_locked_style('num_cohorts') if pls else 0
            nc_label = pls.get_display_label('num_cohorts', "Number of Cohorts") if pls else "Number of Cohorts"
            imgui.set_next_item_width(100)
            changed_nc, new_nc = imgui.slider_int(
                nc_label,
                self.state.sim.num_cohorts,
                1, 144
            )
            if pls and pls.handle_alt_click('num_cohorts'):
                pass  # alt-click intercepted; discard value change
            elif changed_nc:
                self.state.sim.num_cohorts = new_nc
            if pls:
                pls.pop_locked_style(nc_lock_colors)
            self._delayed_tooltip("Each particle is assigned to a cohort. Each cohort shares behavior:\neach cohort has a distinct mutation.")

            imgui.separator()

            # Disable Symmetry
            ds_lock_colors = pls.push_locked_style('DISABLE_SYMMETRY') if pls else 0
            ds_label = pls.get_display_label('DISABLE_SYMMETRY', "Disable Symmetry") if pls else "Disable Symmetry"
            changed_ds, new_ds = imgui.checkbox(
                ds_label,
                self.state.sim.DISABLE_SYMMETRY
            )
            if pls and pls.handle_alt_click('DISABLE_SYMMETRY'):
                pass  # alt-click intercepted; discard value change
            elif changed_ds:
                self.state.sim.DISABLE_SYMMETRY = new_ds
            if pls:
                pls.pop_locked_style(ds_lock_colors)
            self._delayed_tooltip("Allow particles to display \"right / left handed\" behavior,\nleading to clockwise/counterclockwise bias.\nTurn it on to see why we go through trouble\nof calculating \"mirror world\" behavior in entity_update.glsl")

            # Absolute Orientation (combo box with 3 modes)
            combo_items = ["Off", "Y axis", "Radial"]
            ao_lock_colors = pls.push_locked_style('ABSOLUTE_ORIENTATION') if pls else 0
            ao_label = pls.get_display_label('ABSOLUTE_ORIENTATION', "Absolute Orientation") if pls else "Absolute Orientation"
            clicked_ao, new_ao = imgui.combo(
                ao_label,
                self.state.sim.ABSOLUTE_ORIENTATION,
                combo_items
            )
            if pls and pls.handle_alt_click('ABSOLUTE_ORIENTATION'):
                pass  # alt-click intercepted; discard value change
            elif clicked_ao:
                self.state.sim.ABSOLUTE_ORIENTATION = new_ao
            if pls:
                pls.pop_locked_style(ao_lock_colors)
            self._delayed_tooltip("What direction are particles 'facing'? Which way is 'up'?\nOff: use particle velocity\nY axis: align to y axis\nRadial: align to center of canvas")

            # Orientation Mix (only visible if Absolute Orientation != Off)
            if self.state.sim.ABSOLUTE_ORIENTATION != 0:
                om_lock_colors = pls.push_locked_style('ORIENTATION_MIX') if pls else 0
                om_label = pls.get_display_label('ORIENTATION_MIX', "Orientation Mix") if pls else "Orientation Mix"
                imgui.set_next_item_width(100)
                changed_om, new_om = imgui.slider_float(
                    om_label,
                    self.state.sim.ORIENTATION_MIX,
                    0.0, 1.0,
                    "%.2f"
                )
                if pls and pls.handle_alt_click('ORIENTATION_MIX'):
                    pass  # alt-click intercepted; discard value change
                elif changed_om:
                    self.state.sim.ORIENTATION_MIX = new_om
                if pls:
                    pls.pop_locked_style(om_lock_colors)
                self._delayed_tooltip("Blend factor for orientation calculations (0.0 = velocity only, 1.0 = full absolute orientation)")

            imgui.separator()

            # Parameter Sweeps toggle
            _, self.state.sim.parameter_sweeps_enabled = imgui.checkbox(
                "Parameter Sweeps",
                self.state.sim.parameter_sweeps_enabled
            )
            sweep_key = self.keybindings.get_key_display_name('toggle_parameter_sweep')
            self._delayed_tooltip(f"Enable parameter sweeps to vary physics across the canvas.\nPress {sweep_key} to toggle. See Help -> Parameter Sweeps for details.")

        # === Notes Group ===
        imgui.set_next_item_open(self.state.preferences.physics_group_notes)
        notes_open = imgui.collapsing_header("Notes")
        if imgui.is_item_toggled_open():
            self.state.preferences.physics_group_notes = notes_open
        if notes_open:
            imgui.set_next_item_width(-1)
            changed, new_notes = imgui.input_text_multiline(
                "##notes",
                self.state.sim.notes,
                imgui.ImVec2(0, 80),
                imgui.InputTextFlags_.word_wrap | imgui.InputTextFlags_.ctrl_enter_for_new_line
            )
            if changed:
                self.state.sim.notes = new_notes
            self._delayed_tooltip("Optional notes to save with this config.\nThese will be saved when you save the config.\nEnter to finish editing, Ctrl+Enter for newline.")

        imgui.separator()

        # Render the tooltip if window is hovered
        self.render_physics_tooltip()

        imgui.end()

        # Pop sweep preview style colors (pushed before imgui.begin)
        if self.state.sim.sweep_preview_pending_restore:
            imgui.pop_style_color(2)

    def _render_physics_normal_menu_bar(self) -> bool:
        """Render the normal mode menu bar for Physics Settings. Returns whether any menu is open."""
        physics_any_menu_open_this_frame = False
        pls = self.param_lock_service
        if imgui.begin_menu_bar():
            # Track all open menu rectangles separately
            physics_menu_rectangles = []

            # Start with the menu bar itself
            physics_menu_bar_min = imgui.get_window_pos()
            physics_menu_bar_size = imgui.get_window_size()
            physics_menu_rectangles.append((physics_menu_bar_min.x, physics_menu_bar_min.y,
                                           physics_menu_bar_min.x + physics_menu_bar_size.x,
                                           physics_menu_bar_min.y + physics_menu_bar_size.y))

            # Appearance menu
            if imgui.begin_menu("Appearance", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                appearance_settings_menu_min = imgui.get_window_pos()
                appearance_settings_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((appearance_settings_menu_min.x, appearance_settings_menu_min.y,
                                               appearance_settings_menu_min.x + appearance_settings_menu_size.x,
                                               appearance_settings_menu_min.y + appearance_settings_menu_size.y))

                # Color by cohort checkbox
                cbc_lock_colors = pls.push_locked_style('color_by_cohort') if pls else 0
                cbc_label = pls.get_display_label('color_by_cohort', "Color by Cohort") if pls else "Color by Cohort"
                changed_cbc, new_cbc = imgui.checkbox(
                    cbc_label,
                    self.state.sim.color_by_cohort
                )
                if pls and pls.handle_alt_click('color_by_cohort'):
                    pass  # alt-click intercepted; discard value change
                elif changed_cbc:
                    self.state.sim.color_by_cohort = new_cbc
                if pls:
                    pls.pop_locked_style(cbc_lock_colors)
                self._delayed_tooltip("Colors particles based on their cohort assignment\nrather than their behavior.")

                # Hue Sensitivity (only if not color by cohort)
                if not self.state.sim.color_by_cohort:
                    hs_lock_colors = pls.push_locked_style('hue_sensitivity') if pls else 0
                    hs_label = pls.get_display_label('hue_sensitivity', "Hue Sensitivity") if pls else "Hue Sensitivity"
                    changed_hs, new_hs = imgui.slider_float(
                        hs_label, self.state.sim.hue_sensitivity, -1.0, 1.0
                    )
                    if pls and pls.handle_alt_click('hue_sensitivity'):
                        pass  # alt-click intercepted; discard value change
                    elif changed_hs:
                        self.state.sim.hue_sensitivity = new_hs
                    if pls:
                        pls.pop_locked_style(hs_lock_colors)
                    self._delayed_tooltip("Controls color variation based on particle velocity.")

                imgui.separator()

                # Watercolor Mode checkbox (only available in camera views, not field views)
                watercolor_disabled = self.state.sim.current_view_option not in (1, 2)
                if watercolor_disabled:
                    imgui.begin_disabled()
                _, self.state.sim.watercolor_mode = imgui.checkbox(
                    "Watercolor Mode (V)",
                    self.state.sim.watercolor_mode
                )

                # Ink Weight slider (only in watercolor mode, placed right after checkbox)
                if self.state.sim.watercolor_mode:
                    _, self.state.sim.ink_weight = imgui.slider_float(
                        "Ink Weight", self.state.sim.ink_weight, 0.0, 20.0
                    )
                    self._delayed_tooltip("Controls optical density in watercolor mode.\nHigher values = darker/more opaque.")
                self._delayed_tooltip("Enable watercolor rendering effect." + ("\nSwitch to Camera view to enable." if watercolor_disabled else ""))
                if watercolor_disabled:
                    imgui.end_disabled()

                imgui.end_menu()

            # After all menus: check mouse distance from all menu rectangles
            # Find the minimum distance to any rectangle
            if self.physics_menu_bar_has_open_menu and not self.save_popup_open:
                mouse_pos = imgui.get_mouse_pos()

                # Calculate minimum distance to any menu rectangle
                min_distance = float('inf')
                for min_x, min_y, max_x, max_y in physics_menu_rectangles:
                    dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                    dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # If mouse is too far away from all rectangles, signal to close menus
                if min_distance > self.state.preferences.menu_close_threshold:
                    self.force_close_physics_menus = True

            imgui.end_menu_bar()

        # Update physics menu tracking state
        self.physics_menu_bar_has_open_menu = physics_any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_physics_menus and not physics_any_menu_open_this_frame:
            self.force_close_physics_menus = False

        return physics_any_menu_open_this_frame

    def _render_physics_multi_load_mode(self, physics_any_menu_open_this_frame):
        """Render multi-load mode UI for Physics Settings window."""
        # Render multi-load specific UI
        if imgui.begin_menu_bar():
            # Track all open menu rectangles separately
            physics_menu_rectangles = []

            # Start with the menu bar itself
            physics_menu_bar_min = imgui.get_window_pos()
            physics_menu_bar_size = imgui.get_window_size()
            physics_menu_rectangles.append((physics_menu_bar_min.x, physics_menu_bar_min.y,
                                           physics_menu_bar_min.x + physics_menu_bar_size.x,
                                           physics_menu_bar_min.y + physics_menu_bar_size.y))

            if imgui.begin_menu("Multi-Load Settings", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                multi_load_menu_min = imgui.get_window_pos()
                multi_load_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((multi_load_menu_min.x, multi_load_menu_min.y,
                                               multi_load_menu_min.x + multi_load_menu_size.x,
                                               multi_load_menu_min.y + multi_load_menu_size.y))

                # Particle Assignment combo
                assignment_options = ["Random", "Cohorts"]
                current_idx = 0 if self.state.multi_load.assignment_mode == "Random" else 1
                imgui.set_next_item_width(150)
                changed, new_idx = imgui.combo("Particle Assignment", current_idx, assignment_options)
                if changed:
                    self.state.multi_load.assignment_mode = assignment_options[new_idx]
                self._delayed_tooltip("Random: each particle randomly assigned\nCohorts: particles grouped by cohort")

                _, self.state.multi_load.per_config_initial_conditions = imgui.checkbox(
                    "Per-config Initial Conditions", self.state.multi_load.per_config_initial_conditions)
                self._delayed_tooltip("Each config uses its own initial conditions")

                _, self.state.multi_load.per_config_cohorts = imgui.checkbox(
                    "Per-config Cohorts", self.state.multi_load.per_config_cohorts)
                self._delayed_tooltip("Each config uses its own cohort count")

                _, self.state.multi_load.per_config_hazard_rate = imgui.checkbox(
                    "Per-config Hazard Rate", self.state.multi_load.per_config_hazard_rate)
                self._delayed_tooltip("Each config uses its own hazard rate setting")

                imgui.end_menu()

            # Simplified Additional Settings (some items greyed out)
            if imgui.begin_menu("Additional Settings", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                additional_menu_min = imgui.get_window_pos()
                additional_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((additional_menu_min.x, additional_menu_min.y,
                                               additional_menu_min.x + additional_menu_size.x,
                                               additional_menu_min.y + additional_menu_size.y))

                # Boundary Conditions
                boundary_options = ["Bounce", "Reset", "Wrap"]
                imgui.set_next_item_width(100)
                if imgui.begin_combo("Boundary Conditions", boundary_options[self.state.sim.boundary_conditions]):
                    for i, option in enumerate(boundary_options):
                        if imgui.selectable(option, self.state.sim.boundary_conditions == i)[0]:
                            self.state.sim.boundary_conditions = i
                    imgui.end_combo()

                # Initial Conditions (greyed if per-config)
                if self.state.multi_load.per_config_initial_conditions:
                    imgui.begin_disabled()
                initial_options = ["Grid", "Random", "Ring"]
                imgui.set_next_item_width(100)
                if imgui.begin_combo("Initial Conditions", initial_options[self.state.sim.initial_conditions]):
                    for i, option in enumerate(initial_options):
                        if imgui.selectable(option, self.state.sim.initial_conditions == i)[0]:
                            self.state.sim.initial_conditions = i
                    imgui.end_combo()
                if self.state.multi_load.per_config_initial_conditions:
                    imgui.end_disabled()

                # Cohorts (greyed if per-config)
                if self.state.multi_load.per_config_cohorts:
                    imgui.begin_disabled()
                imgui.set_next_item_width(100)
                _, self.state.sim.num_cohorts = imgui.slider_int("Number of Cohorts", self.state.sim.num_cohorts, 1, 144)
                if self.state.multi_load.per_config_cohorts:
                    imgui.end_disabled()

                # Hazard Rate (conditional on per-config setting)
                if self.state.multi_load.per_config_hazard_rate:
                    imgui.begin_disabled()
                # Power-scaled slider for fine control at low values while reaching 0.0
                HAZARD_MAX = 0.05
                HAZARD_POWER = 3.0  # Higher = more resolution at low end
                # Convert actual value to slider position (0-1)
                slider_pos = (self.state.sim.HAZARD_RATE / HAZARD_MAX) ** (1.0 / HAZARD_POWER)
                _, new_pos = imgui.slider_float(
                    "Hazard Rate",
                    slider_pos,
                    0.0,
                    1.0,
                    f"{self.state.sim.HAZARD_RATE:.5f}"
                )
                # Convert slider position back to actual value
                self.state.sim.HAZARD_RATE = HAZARD_MAX * (new_pos ** HAZARD_POWER)
                if self.state.multi_load.per_config_hazard_rate:
                    imgui.end_disabled()
                self._delayed_tooltip("Probability per frame that particles reset to initial conditions")

                # Disable these options in multi-load (per-config settings)
                imgui.begin_disabled()
                imgui.checkbox("Disable Symmetry", False)
                imgui.checkbox("Absolute Orientation", False)
                imgui.end_disabled()
                self._delayed_tooltip("Per-config settings in Multi-Load mode")

                # Parameter Sweeps (disabled)
                imgui.begin_disabled()
                imgui.checkbox("Parameter Sweeps", False)
                imgui.end_disabled()
                self._delayed_tooltip("Disabled in Multi-Load mode")

                imgui.end_menu()

            # Appearance (unchanged, copy from normal mode)
            if imgui.begin_menu("Appearance", not self.force_close_physics_menus):
                physics_any_menu_open_this_frame = True
                # Add this menu's bounding box to the list
                appearance_menu_min = imgui.get_window_pos()
                appearance_menu_size = imgui.get_window_size()
                physics_menu_rectangles.append((appearance_menu_min.x, appearance_menu_min.y,
                                               appearance_menu_min.x + appearance_menu_size.x,
                                               appearance_menu_min.y + appearance_menu_size.y))

                _, self.state.sim.color_by_cohort = imgui.checkbox("Color by Cohort", self.state.sim.color_by_cohort)
                if not self.state.sim.color_by_cohort:
                    imgui.set_next_item_width(100)
                    _, self.state.sim.hue_sensitivity = imgui.slider_float("Hue Sensitivity", self.state.sim.hue_sensitivity, -1.0, 1.0)
                watercolor_disabled_adv = self.state.sim.current_view_option not in (1, 2)
                if watercolor_disabled_adv:
                    imgui.begin_disabled()
                _, self.state.sim.watercolor_mode = imgui.checkbox("Watercolor Mode", self.state.sim.watercolor_mode)
                if watercolor_disabled_adv:
                    imgui.end_disabled()
                if self.state.sim.watercolor_mode:
                    imgui.set_next_item_width(100)
                    _, self.state.sim.ink_weight = imgui.slider_float("Ink Weight", self.state.sim.ink_weight, 0.0, 4.0)
                imgui.end_menu()

            # After all menus: check mouse distance from all menu rectangles
            # Find the minimum distance to any rectangle
            if self.physics_menu_bar_has_open_menu and not self.save_popup_open:
                mouse_pos = imgui.get_mouse_pos()

                # Calculate minimum distance to any menu rectangle
                min_distance = float('inf')
                for min_x, min_y, max_x, max_y in physics_menu_rectangles:
                    dx = max(min_x - mouse_pos.x, 0, mouse_pos.x - max_x)
                    dy = max(min_y - mouse_pos.y, 0, mouse_pos.y - max_y)
                    distance = (dx * dx + dy * dy) ** 0.5
                    min_distance = min(min_distance, distance)

                # If mouse is too far away from all rectangles, signal to close menus
                if min_distance > self.state.preferences.menu_close_threshold:
                    self.force_close_physics_menus = True

            imgui.end_menu_bar()

        # Update physics menu tracking state
        self.physics_menu_bar_has_open_menu = physics_any_menu_open_this_frame
        # Reset force close flag after processing
        if self.force_close_physics_menus and not physics_any_menu_open_this_frame:
            self.force_close_physics_menus = False

        # Multi-load controls
        imgui.text("Multi-Load Controls")
        imgui.separator()

        # Get config count from service
        config_count = self.multi_load_service.get_config_count() if self.multi_load_service else 0

        _, self.state.multi_load.simultaneous_configs = imgui.slider_float(
            "Simultaneous Configs", self.state.multi_load.simultaneous_configs, 0.0, float(max(1, config_count-.001)))
        self._delayed_tooltip("Sets the size of the sliding window that blends the configs.\nAt 0 there will be no blending, just one config at a time.\nAt max value, all loaded configs will be active at once.")
        _, self.state.multi_load.progression_pace = imgui.slider_float(
            "Progression Pace", self.state.multi_load.progression_pace, 0.0, 1.0)
        self._delayed_tooltip("Defines the pace at which we animate current\nprogress through the loaded configs.")

        # Sync current progress from service (for auto-advancement display)
        if self.multi_load_service:
            current_progress_value = self.multi_load_service.current_progress
        else:
            current_progress_value = self.state.multi_load.current_progress

        changed, new_progress = imgui.slider_float(
            "Current Progress", current_progress_value, 0.0, 1.0)
        if changed and self.multi_load_service:
            # User manually changed the slider - update service directly
            self.multi_load_service.set_progress(new_progress)
        # Always sync state from service for next frame
        if self.multi_load_service:
            self.state.multi_load.current_progress = self.multi_load_service.current_progress
        self._delayed_tooltip("Determines where in the config lineup we are.")

        imgui.separator()
        if imgui.button("Import from Config Clipboard"):
            self._request_import_clipboard_to_multiload = True
        self._delayed_tooltip("Replace the loaded configs with the contents\nof the config clipboard (see Extras).")
        imgui.separator()
        imgui.text(f"Loaded Configurations ({config_count}/64)")
        imgui.separator()

        if config_count == 0:
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "No configs loaded")
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Use File -> Load to add")
        else:
            # Render config list with remove buttons
            for i in range(config_count):
                filename = self.multi_load_service.get_filename(i)
                if filename:
                    # Config name
                    imgui.text(f"{i+1}. {filename}")
                    imgui.same_line()
                    # Remove button (aligned to right)
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
                    if imgui.small_button(f"Remove##{i}"):
                        self.multi_load_service.remove_config(i)
                    imgui.pop_style_color(2)

        imgui.end()
        if self.state.sim.sweep_preview_pending_restore:
            imgui.pop_style_color(2)
