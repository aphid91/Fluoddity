"""Preferences window: world size, physics frequency, mouse mode, view, appearance."""
from imgui_bundle import imgui

# Aspect ratio options: (label, ratio_string, is_separator)
_ASPECT_RATIO_OPTIONS = [
    ("1:1",  "1:1",  False),
    ("4:3",  "4:3",  False),
    ("16:9", "16:9", False),
    ("2:1",  "2:1",  False),
    ("3:1",  "3:1",  False),
    ("---",  None,   True),   # Separator (not selectable)
    ("3:4",  "3:4",  False),
    ("9:16", "9:16", False),
    ("1:2",  "1:2",  False),
    ("1:3",  "1:3",  False),
]


class PreferencesWindowMixin:
    """Mixin for preferences window. Combined into UI via multiple inheritance."""

    def render_preferences_window(self):
        """Render the Preferences window (closeable)."""
        recording_active = self._display_info.get('recording_active', False)
        video_pending = self._display_info.get('video_pending', False)

        # Apply red tint to window background when recording or pending
        if recording_active or video_pending:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

        # Use p_open to allow closing with X button
        expanded, self.state.preferences.show_preferences_window = imgui.begin("Preferences", True)

        if expanded:
            # === World Size section ===
            imgui.text("World Size")

            # Use input_float - only apply when user commits (Enter or focus loss)
            changed, new_value = imgui.input_float(
                "World Size",
                self.state.preferences.world_size,
                step=0.0,  # No step buttons
                step_fast=0.0,
                format="%.2f"
            )

            # Clamp to valid range
            if new_value < 0.02:
                new_value = 0.02
            elif new_value > 4.0:
                new_value = 4.0

            # Update the displayed value (clamping happens immediately)
            self.state.preferences.world_size = new_value

            # Only trigger world size change when user commits the edit
            if imgui.is_item_deactivated_after_edit():
                if abs(self.state.preferences.world_size - self._last_applied_world_size) > 0.001:
                    self._request_world_size_change = True

            self._delayed_tooltip("EXPENSIVE - Controls the size of the simulation world.\nAffects both entity count and canvas resolution to keep density ~fixed")

            # Canvas Aspect Ratio dropdown
            current_ratio = self.state.preferences.canvas_aspect_ratio
            if imgui.begin_combo("World Shape", current_ratio):
                for label, ratio_str, is_sep in _ASPECT_RATIO_OPTIONS:
                    if is_sep:
                        imgui.separator()
                    else:
                        selected = (ratio_str == current_ratio)
                        clicked, _ = imgui.selectable(label, selected)
                        if clicked and ratio_str != current_ratio:
                            self.state.preferences.canvas_aspect_ratio = ratio_str
                            self._request_world_size_change = True
                            self._request_reload = True
                        if selected:
                            imgui.set_item_default_focus()
                imgui.end_combo()
            self._delayed_tooltip("Changes the canvas aspect ratio.")

            imgui.separator()

            # === Physics Update Frequency section ===
            imgui.text("Physics Update Frequency")

            # Lock speedmult to motion_blur_samples when recording video
            if recording_active:
                locked_value = self.state.preferences.motion_blur_samples
                imgui.begin_disabled()
                imgui.slider_int(
                    label="Rate",
                    v=locked_value,
                    v_min=1,
                    v_max=6,
                    format=f"x{locked_value} ({locked_value * 60}hz) [locked]"
                )
                imgui.end_disabled()
            else:
                # Slider with custom format showing multiplier and hz
                current_hz = self.state.preferences.speedmult * 60
                _, self.state.preferences.speedmult = imgui.slider_int(
                    label="Rate",
                    v=self.state.preferences.speedmult,
                    v_min=1,
                    v_max=30,
                    format=f"x%d ({current_hz}hz)"
                )
            self._delayed_tooltip("EXPENSIVE- Multiple physics steps can be calculated each\nrender frame and blended together for faster physics.\nMotion blur can be costly for high frequencies,\ntry turning it off if things feel sluggish.")

            # Motion blur checkbox (lock during recording)
            if recording_active:
                imgui.begin_disabled()

            _, self.state.preferences.motion_blur = imgui.checkbox(
                "Motion Blur",
                self.state.preferences.motion_blur
            )
            self._delayed_tooltip("EXPENSIVE- Multiple physics steps can be calculated each\nrender frame and blended together for faster physics.\nMotion blur can be costly for high frequencies,\ntry turning it off if things feel sluggish.")

            if recording_active:
                imgui.end_disabled()

            # Blur Quality slider (only shown when motion blur is enabled)
            if self.state.preferences.motion_blur:
                imgui.indent(20)
                # Custom format for blur quality
                blur_val = self.state.preferences.blur_quality
                if blur_val == 1:
                    blur_format = "1 : Every Frame"
                else:
                    blur_format = f"{blur_val} : Every {blur_val} Frames"

                _, self.state.preferences.blur_quality = imgui.slider_int(
                    "Blur Quality",
                    self.state.preferences.blur_quality,
                    1, 20,
                    format=blur_format
                )
                self._delayed_tooltip("Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance")
                imgui.unindent(20)

            imgui.separator()

            # === Mouse Interaction section ===
            imgui.text("Mouse Interaction (Press 'T' to toggle)")

            # Mouse mode combo box (locked when multi-load enabled)
            if self.state.multi_load.multi_load_enabled:
                imgui.begin_disabled()
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Mouse Mode: Draw Trail (locked in Multi-Load)")
                imgui.end_disabled()
            else:
                mouse_modes = ["Select Particle", "Draw Trail"]
                current_mode_idx = mouse_modes.index(self.state.preferences.mouse_mode) if self.state.preferences.mouse_mode in mouse_modes else 0
                clicked, new_mode_idx = imgui.combo("Mouse Mode", current_mode_idx, mouse_modes)
                if clicked:
                    self.state.preferences.mouse_mode = mouse_modes[new_mode_idx]
                self._delayed_tooltip("In select Particle mode, clicking selects a particle rule to focus on.\nIn Draw trail mode, click and drag to leave trails on the canvas.\nSee Help->Controls for more")

            # Draw mode sliders (only show when in Draw Trail mode)
            if self.state.preferences.mouse_mode == "Draw Trail":
                imgui.indent(20)
                _, self.state.preferences.draw_size = imgui.slider_float(
                    "Draw Size",
                    self.state.preferences.draw_size,
                    0.01, 0.5,
                    format="%.3f"
                )
                _, self.state.preferences.draw_power = imgui.slider_float(
                    "Draw Power",
                    self.state.preferences.draw_power,
                    0.1, 5.0,
                    format="%.2f"
                )
                imgui.unindent(20)

            imgui.separator()

            # === View section ===
            imgui.text("View")

            # View dropdown - conditionally show Force/Strafe field views
            view_items = self.view_option_labels + ['Camera (Particles rendered as dots)', 'Camera [Tiled]']
            adp = getattr(self, 'advanced_drawing_processor', None)
            if adp is not None and adp.field_texture is not None:
                view_items = view_items + ['Force Field', 'Strafe Field']

            changed, self.state.sim.current_view_option = imgui.combo(
                label="Current View",
                current_item=self.state.sim.current_view_option,
                items=view_items
            )

            if changed:
                # cam_brush_mode is True for Camera (index 1) and Tiled (index 2)
                # Force Field (3) and Strafe Field (4) are raw texture views like canvas
                if self.state.sim.current_view_option in (1, 2):
                    self.state.camera.cam_brush_mode = True
                else:
                    self.state.camera.cam_brush_mode = False
                    # Watercolor is unsupported in non-camera views
                    self.state.sim.watercolor_mode = False

            # Physics tooltips checkbox
            _, self.state.preferences.physics_tooltips_enabled = imgui.checkbox(
                "Physics Tooltips",
                self.state.preferences.physics_tooltips_enabled
            )
            self._delayed_tooltip("Enable verbose tooltip and vector diagram for physics sliders.")

            # Arrow debug checkbox - label changes when advanced drawing is open
            arrow_label = ("View Draw Target Arrows"
                           if self.state.preferences.advanced_drawing_enabled
                           else "View Trail Arrows")
            _, self.state.preferences.debug_arrows = imgui.checkbox(
                arrow_label,
                self.state.preferences.debug_arrows
            )
            self._delayed_tooltip("Render a grid of arrows to help visualize the active draw target's vector field.")

            # Arrow sensitivity slider (only show when debug arrows enabled)
            if self.state.preferences.debug_arrows:
                imgui.indent(20)
                _, self.state.preferences.arrow_sensitivity = imgui.slider_float(
                    "Arrow Sensitivity",
                    self.state.preferences.arrow_sensitivity,
                    1.0, 20.0,
                    format="%.1f"
                )
                imgui.unindent(20)

            imgui.separator()

            # === Appearance section ===
            imgui.text("Appearance")

            # Brightness slider
            _, self.state.preferences.brightness = imgui.slider_float(
                "Brightness",
                self.state.preferences.brightness,
                0.01, 10.0,
                format="%.2f"
            )
            self._delayed_tooltip("Global brightness multiplier for the output.")

            # Tonemap Softness slider
            _, self.state.preferences.tonemap_softness = imgui.slider_float(
                "Tonemap Softness",
                self.state.preferences.tonemap_softness,
                0.1, 5.0,
                format="%.2f"
            )
            self._delayed_tooltip("Controls highlight compression (asinh stretch).\nLow values = more linear (brighter highlights).\nHigh values = more logarithmic (reveals faint detail).")

            # Exposure / Cheap Blur slider
            _, self.state.preferences.exposure = imgui.slider_float(
                "Exposure / Cheap Blur",
                self.state.preferences.exposure,
                0.0, 1.0,
                format="%.2f"
            )
            self._delayed_tooltip("Blend frames together for a cheap motion blur or set near 1 for a long exposure effect.")

            # Bloom checkbox + sliders (disabled in watercolor mode)
            watercolor_active = self.state.sim.watercolor_mode
            if watercolor_active:
                imgui.begin_disabled()
            _, self.state.preferences.bloom_enabled = imgui.checkbox(
                "Bloom",
                self.state.preferences.bloom_enabled
            )
            if watercolor_active:
                self._delayed_tooltip("Bloom is disabled in Watercolor mode.")
            else:
                self._delayed_tooltip("Add a glow effect around bright areas.")

            if self.state.preferences.bloom_enabled and not watercolor_active:
                imgui.indent(20)
                _, self.state.preferences.bloom_threshold = imgui.slider_float(
                    "Threshold",
                    self.state.preferences.bloom_threshold,
                    0.0, 2.0,
                    format="%.2f"
                )
                self._delayed_tooltip("Brightness cutoff for bloom extraction.\nLower = more glow everywhere.")

                _, self.state.preferences.bloom_intensity = imgui.slider_float(
                    "Intensity",
                    self.state.preferences.bloom_intensity,
                    0.0, 3.0,
                    format="%.2f"
                )
                self._delayed_tooltip("Strength of the bloom glow.")

                _, self.state.preferences.bloom_radius = imgui.slider_float(
                    "Radius",
                    self.state.preferences.bloom_radius,
                    0.1, 3.0,
                    format="%.2f"
                )
                self._delayed_tooltip("Spread of the bloom blur kernel.")
                imgui.unindent(20)
            if watercolor_active:
                imgui.end_disabled()

            # === Performance readout ===
            imgui.separator()
            fps = imgui.get_io().framerate
            ms = 1000.0 / fps if fps > 0 else 0.0
            imgui.text(f"Performance: {fps:.1f} FPS ({ms:.2f} ms/frame)")

        imgui.end()

        # Restore normal window background color if it was changed
        if recording_active or video_pending:
            imgui.pop_style_color()

    def _get_key_combo(self, action: str, modifier: str = "") -> str:
        """
        Get a formatted key combination string for display.

        Args:
            action: The action name from keyboard_controls.json
            modifier: Optional modifier like "Ctrl+" or "Shift+"

        Returns:
            Formatted string like "Ctrl+C" or "WASD"
        """
        key = self.keybindings.get_key_display_name(action)
        if modifier:
            return f"{modifier}{key}"
        return key
