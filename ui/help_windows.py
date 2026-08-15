"""Help and informational windows: Controls, Tutorial, Parameter Sweeps, Performance, Video Recording."""
from imgui_bundle import imgui


class HelpWindowsMixin:
    """Mixin for help/informational windows. Combined into UI via multiple inheritance."""

    def render_controls_window(self):
        """Render the Controls help window (closeable)."""
        expanded, self.state.preferences.show_controls_window = imgui.begin("Controls", True)

        if expanded:            
            imgui.text("Mouse Controls")
            imgui.separator()

            imgui.text("Select Particle mode:")
            imgui.indent(20)
            imgui.bullet_text("Left click - select particle and adopt its rule (with mutations)")
            imgui.bullet_text("Right click - undo")
            imgui.bullet_text("You can undo particle selection and randomize actions.")
            imgui.unindent(20)

            imgui.text("Draw Trail mode:")
            imgui.indent(20)
            imgui.bullet_text("Click and drag - Draw trails on the canvas")
            imgui.bullet_text("Right click and drag - Erase trails on the canvas")
            imgui.unindent(20)

            imgui.spacing()
            imgui.text("Keyboard Controls")
            imgui.separator()
            imgui.text("Edit 'keyboard_controls.json' in documents/Fluoddity")
            # Camera movement keys
            w = self.keybindings.get_key_display_name("camera_forward")
            a = self.keybindings.get_key_display_name("camera_left")
            s = self.keybindings.get_key_display_name("camera_backward")
            d = self.keybindings.get_key_display_name("camera_right")
            imgui.bullet_text(f"{w}{a}{s}{d} - Move camera")

            q = self.keybindings.get_key_display_name("camera_out")
            e = self.keybindings.get_key_display_name("camera_in")
            imgui.bullet_text(f"{q}/{e} - Zoom out/in")

            imgui.bullet_text(f"Ctrl+{self.keybindings.get_key_display_name('copy_config_with_ctrl')} - Copy config to clipboard")
            imgui.bullet_text(f"Ctrl+{self.keybindings.get_key_display_name('paste_config_with_ctrl')} - Paste config from clipboard")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('randomize_mutations')} - Randomize Mutation seed")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('exit_keybinding')} - Exit application")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('record_screen')} - Toggle video recording")
            imgui.bullet_text("Shift+P - Take screenshot")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_pause')} - Pause/resume simulation")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('reset_keybinding')} - Reset particles to Initial Conditions")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_parameter_sweep')} - Toggle parameter sweeps")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_watercolor')} - Toggle watercolor mode")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('reload_shaders')} - Reload shaders (Sometimes fixes frozen/black screen)")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_help')} - Show tutorial")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('toggle_mouse_mode')} - Toggle mouse mode: Selection/Trail Drawing")
            imgui.bullet_text(f"{self.keybindings.get_key_display_name('randomize_rules')} - Randomize particle behavior + new mutation seed")
            
            imgui.spacing()
            imgui.text("Slider Tips")
            imgui.separator()
            imgui.bullet_text("Right-click slider - Context menu to adjust range\n (context menu only for Basics/Forces/Advanced)")
            imgui.bullet_text("Ctrl+click slider - Enter custom value directly")


        imgui.end()

    def render_parameter_sweeps_window(self):
        """Render the Parameter Sweeps help window (closeable)."""
        expanded, self.state.preferences.show_parameter_sweeps_window = imgui.begin("Parameter Sweeps", True)

        if expanded:
            imgui.text_wrapped(
                "Parameter sweeps let you vary physics settings across the screen, "
                "creating a gradient where each position uses different parameter values."
            )

            imgui.spacing()
            imgui.text("How to Use")
            imgui.separator()

            sweep_key = self.keybindings.get_key_display_name('toggle_parameter_sweep')
            imgui.bullet_text(f"Enable sweeps: Additional Settings -> Parameter Sweeps (or press {sweep_key})")
            imgui.bullet_text("Each parameter can sweep on X-axis, Y-axis, or by Cohort")
            imgui.bullet_text("The swept parameter will vary from slider_min to slider_max")
            imgui.bullet_text("Each slider has up/down buttons to its left which\nwiden/narrow the slider range.")
            imgui.bullet_text("Right click on sliders to manually set ranges")
            imgui.bullet_text("When using X and or Y sweeps, click anywhere on the canvas to\nset slider values. Then when you turn sweeps off,\neverywhere will behave like the region you clicked. ")
            imgui.bullet_text("Right click will temporarily disable sweeps allowing you to see\nthe effects of your slider values. Click anywhere to end the 'preview'.")

            imgui.spacing()
            imgui.text("Sweep Directions")
            imgui.separator()

            imgui.bullet_text("Left click for Normal, Right click for Inverse")
            imgui.bullet_text("Normal (->): Left/bottom = min, Right/top = max")
            imgui.bullet_text("Inverse (<-): Left/bottom = max, Right/top = min")
            imgui.bullet_text("Off: Parameter uses its slider value everywhere")

            imgui.spacing()
            imgui.text("Tips")
            imgui.separator()
            imgui.bullet_text("Some sliders have hard capped ranges, others are unbounded")
            imgui.text_wrapped(
                "Cohort sweeps vary parameters across particle groups rather than "
                "screen position."
            )
            imgui.spacing()
            imgui.text_wrapped(
                "Combine X and Y sweeps on different parameters to explore "
                "2D parameter spaces. For example, sweep Drag on X and "
                "Sensor Angle on Y to see how they interact. "
            )

        imgui.end()

    def render_tutorial_window(self):
        """Render the Tutorial help window (closeable)."""
        expanded, self.state.preferences.show_tutorial_window = imgui.begin("Tutorial", True)

        if expanded:
            imgui.text_wrapped(
                "Fluoddity is like an interactive lava lamp. "
                "Thousands of particles interact resulting in a great variety of forms and patterns. "
                "Thumb through the presets in File->Load to see what is possible."
            )

            imgui.spacing()
            if imgui.collapsing_header("Basics", imgui.TreeNodeFlags_.default_open):
                imgui.text_wrapped("(see help->Controls for more)")
                # Camera movement keys
                w = self.keybindings.get_key_display_name("camera_forward")
                a = self.keybindings.get_key_display_name("camera_left")
                s = self.keybindings.get_key_display_name("camera_backward")
                d = self.keybindings.get_key_display_name("camera_right")
                imgui.bullet_text(f"Move the camera around with {w}{a}{s}{d}.")
                q = self.keybindings.get_key_display_name("camera_out")
                e = self.keybindings.get_key_display_name("camera_in")
                imgui.bullet_text(f"Zoom in or out with {q}/{e} or scroll wheel.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('reset_keybinding')} to reset the simulation.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_pause')} to toggle pause.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('randomize_mutations')} for a fresh crop of mutations.")
                imgui.bullet_text("Click to draw trails or select particles.")
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_mouse_mode')} to toggle between drawing and selecting.")
                imgui.separator_text("Trail Drawing Mode - White reticle visible")
                imgui.bullet_text("Click and drag to draw trails")
                imgui.bullet_text("Right Click and drag to erase trails")
                imgui.separator_text("Particle Selection Mode - no reticle")
                imgui.bullet_text("Click a particle to select it and other\nparticles will copy its behavior (with mutations)")
                imgui.bullet_text("Right click to go back and undo particle selection")
                imgui.bullet_text(f"Right click also undos Randomize actions ({self.keybindings.get_key_display_name('randomize_mutations')}/{self.keybindings.get_key_display_name('randomize_rules')})")
                imgui.separator()
                imgui.bullet_text(f"Press {self.keybindings.get_key_display_name('toggle_help')} to toggle this Help window.")

            imgui.spacing()
            if imgui.collapsing_header("Rules"):
                imgui.text_wrapped(
                    "There is no fixed particle behavior in Fluoddity. "
                    "Each particle has a 'Rule' which determines how it moves in response to nearby trails. "
                    "These rules can be mutated and evolved. In particle selection mouse mode, click on a particle to set it's rule as the 'active Rule'. "
                    "Now every particle will adopt that rule (with mutations). right click to undo setting a new target rule."
                )
            imgui.spacing()
            if imgui.collapsing_header("Save/Load"):
                copy_key = self.keybindings.get_key_display_name('copy_config_with_ctrl')
                paste_key = self.keybindings.get_key_display_name('paste_config_with_ctrl')
                imgui.text_wrapped(
                    "Create something you like? Save it as a new preset with File->Save.\nIt will be saved to Documents/Fluoddity/physics_configs. "
                    "The active rule, current mutations, and everything on the physics panel will be restored when you load the save (Physics Sliders, Additional Settings, Appearance, and Notes) "
                    f"\nYou can also press Ctrl-{copy_key} to copy a 'save string' to your clipboard, and Ctrl-{paste_key} to load a save string from the clipboard. "
                    "\nSave strings are just text copied your clipboard (typically a couple thousand characters), so they can be easily shared or stashed."
                )

            imgui.spacing()
            if imgui.collapsing_header("Trails"):
                imgui.text_wrapped(
                    "Particles in Fluoddity can't directly 'see' each other. "
                    "Instead, they interact by leaving pheremone trails as they move, like ants. "
                    "These trails accumulate on the 'Canvas' where particles can see them. Trails spread out and fade over time. "
                    "You can try writing your own pheremone trails to the canvas with 'Draw Trails' mouse mode (Preferences -> mouse mode)"
                )



            imgui.spacing()
            if imgui.collapsing_header("Mutations"):
                randomize_key = self.keybindings.get_key_display_name('randomize_rules')
                seed_key =  self.keybindings.get_key_display_name('randomize_mutations')
                imgui.text_wrapped(
                    "Particles are grouped into 'Cohorts'. Each cohort shares a single mutation, so all the particles in a given cohort behave the same. "
                    "When the Mutation Scale slider is greater than 0, different cohorts can behave differently, sometimes radically so. "
                    "The new rules generated by these mutations can also be selected as the active rule, so you can evolve particle behavior over many iterations. "
                )
                imgui.bullet_text(f"If you want to reset all the cohorts to totally random Rules,\npress {randomize_key}. (this can be undone with right click)")
                imgui.bullet_text(f"If you want to see a fresh set of mutations with the same\nbase Rule, press {seed_key}. (this can be undone with right click)")

            imgui.spacing()
            if imgui.collapsing_header("Sliders"):
                imgui.text_wrapped(
                    "All sliders support Ctrl click to enter custom values. You can exceed slider range this way, "
                    "but it isn't always advisable. The primary parameter sliders:'Basic', 'Forces', and 'Advanced'  are special. Right click on them to set "
                    "custom ranges or add random jitter with the context menu. You can also vary their value across the canvas with parameter sweeps. See Help -> Parameter Sweeps for more."
                )

        imgui.end()

    def render_performance_window(self):
        """Render the Performance help window (closeable)."""
        expanded, self.state.preferences.show_performance_window = imgui.begin("Performance", True)

        if expanded:
            imgui.text_wrapped(
                "The options for World size, Physics update Frequency, and motion blur "
                "can significantly affect performance. World size and update frequency "
                "trade against each other so if you double one, halve the other for similar performance."
                "Motion blur gets more expensive with large worldsizes and high frequencies."
            )

            imgui.spacing()
            imgui.text("Example Setups")
            imgui.separator()

            imgui.bullet_text("x20 physics frequency with worldsize 0.5, motion blur every 5 frames")
            imgui.bullet_text("x9 physics frequency with worldsize 1.0, motion blur every 3 frames")

            imgui.spacing()
            imgui.text_wrapped(
                "These run well on my 5060 ."
            )

        imgui.end()

    def render_video_recording_window(self):
        """Render the Screen Recording controls window (closeable)."""
        recording_active = self._display_info.get('recording_active', False)
        video_pending = self._display_info.get('video_pending', False)
        scheduled_start_frame = self._display_info.get('video_scheduled_start_frame', 0)

        # Apply red tint when recording or pending
        if recording_active or video_pending:
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

        expanded, self.show_video_recording_window = imgui.begin("Screen Recording", True)

        if expanded:
            record_key = self.keybindings.get_key_display_name('record_screen')
            if recording_active:
                imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "RECORDING IN PROGRESS")
                imgui.text(f"Press {record_key} to stop recording")
                imgui.separator()
            elif video_pending:
                current_frame = self._display_info.get('frame_count', 0)
                imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.3, 1.0), "WAITING FOR START FRAME")
                imgui.text(f"Recording starts at frame {scheduled_start_frame}")
                imgui.text(f"Frames remaining: {scheduled_start_frame - current_frame}")
                imgui.text(f"Press {record_key} to cancel")
                imgui.separator()

            imgui.text(f"Press {record_key} to start/stop video recording")
            imgui.text(f"Press Shift+{record_key} to take a screenshot")
            imgui.spacing()

            # Current frame count display
            current_frame = self._display_info.get('frame_count', 0)
            imgui.text(f"Current Frame: {current_frame}")
            imgui.spacing()

            # Video End Frame input
            _, self.state.preferences.video_end_frame = imgui.input_int(
                'Video End Frame',
                self.state.preferences.video_end_frame
            )
            self._delayed_tooltip("Target frame for video to end on.\nWhen set, recording will be delayed until the\ncalculated start frame is reached.\nSet to 0 to start recording immediately.")
            imgui.spacing()

            # Video Length (in seconds) - converts to/from max_frames internally
            video_length_seconds = self.state.preferences.max_frames / 60.0
            changed, new_length = imgui.drag_float(
                'Video Length',
                video_length_seconds,
                v_speed=0.5,
                v_min=1.0,
                v_max=300.0,
                format="%.0f seconds"
            )
            if changed:
                self.state.preferences.max_frames = int(new_length * 60)
            self._delayed_tooltip("After Video reaches this length, the recording will be stopped")

            imgui.separator()

            # Streamline / audio capture. Locked while recording: both change
            # what the pipeline is doing mid-render.
            if recording_active:
                imgui.begin_disabled()

            _, self.state.audio.record_audio = imgui.checkbox(
                "Enable Audio", self.state.audio.record_audio
            )
            self._delayed_tooltip(
                "Render the streamline voice into the video's audio track.\n"
                "Generated per frame rather than in realtime, so it stays in\n"
                "sync however slowly the render runs. Live audio playback is\n"
                "suspended for the duration."
            )
            if self.state.audio.record_audio:
                sr = self.state.audio.sample_rate
                imgui.text_disabled(f"  {sr} Hz, {sr / 60.0:.0f} samples/frame")

            _, self.state.streamline.render_to_video = imgui.checkbox(
                "Streamlines Render To Video", self.state.streamline.render_to_video
            )
            self._delayed_tooltip(
                "Composite the streamline overlay into recorded frames.\n"
                "Requires the Streamlines overlay to be enabled."
            )
            if (self.state.streamline.render_to_video
                    and not self.state.streamline.enabled):
                imgui.text_disabled("  (Streamlines overlay is off)")

            if recording_active:
                imgui.end_disabled()

            imgui.separator()

            # Lock motion_blur_samples during recording
            if recording_active:
                imgui.begin_disabled()

            # Capture Physics Frequency / Screenshot samples
            current_hz = self.state.preferences.motion_blur_samples * 60
            _, self.state.preferences.motion_blur_samples = imgui.slider_int(
                'Capture Physics Frequency',
                self.state.preferences.motion_blur_samples,
                v_min=1,
                v_max=100,
                format=f"x%d ({current_hz}hz)"
            )
            self._delayed_tooltip("Physics steps per frame for video/screenshots.\nHigher values = faster physics with smoother motion blur.\nAlso determines screenshot exposure (# of samples to blend together).")

            if recording_active:
                imgui.end_disabled()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.8, 0.0, 1.0),
                    "(Locked during recording)"
                )

            # Frame range display
            imgui.spacing()
            total_sim_frames = self.state.preferences.max_frames * self.state.preferences.motion_blur_samples
            video_end_frame = self.state.preferences.video_end_frame
            if video_end_frame > 0 and video_end_frame - total_sim_frames >= current_frame:
                start_frame = video_end_frame - total_sim_frames
                end_frame = video_end_frame
            else:
                start_frame = current_frame
                end_frame = current_frame + total_sim_frames
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
                f"Frame Range: {start_frame} --- {end_frame}"
            )
            self._delayed_tooltip(f"Estimated recording range based on current settings.\nTotal simulation frames: {total_sim_frames}\n({self.state.preferences.max_frames} output frames x {self.state.preferences.motion_blur_samples} physics steps)")
            imgui.spacing()

            # Motion Blur checkbox (overrides preferences during recording)
            _, self.state.preferences.recording_motion_blur = imgui.checkbox(
                "Motion Blur (Recording)",
                self.state.preferences.recording_motion_blur
            )
            self._delayed_tooltip("Enable motion blur during video recording.\nThis setting overrides the Motion Blur checkbox in Preferences while recording.")

            # Blur Quality slider (only shown when recording motion blur is enabled)
            if self.state.preferences.recording_motion_blur:
                imgui.indent(20)
                # Custom format for blur quality
                blur_val = self.state.preferences.recording_blur_quality
                if blur_val == 1:
                    blur_format = "1 : Every Frame"
                else:
                    blur_format = f"{blur_val} : Every {blur_val} Frames"

                _, self.state.preferences.recording_blur_quality = imgui.slider_int(
                    "Blur Quality (Recording)",
                    self.state.preferences.recording_blur_quality,
                    1, 20,
                    format=blur_format
                )
                self._delayed_tooltip("Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance.\nThis setting overrides the Blur Quality slider in Preferences while recording.")
                imgui.unindent(20)

            # Downsample Resolution Factor (was Supersample Kernel Width)
            _, self.state.preferences.supersample_k = imgui.input_int('Downsample Resolution Factor', self.state.preferences.supersample_k)
            self._delayed_tooltip("Set to '2' to render a video at half resolution.")

            # Filename input
            _, self.state.preferences.filename_prefix = imgui.input_text(
                'Filename',
                self.state.preferences.filename_prefix,
                256
            )
            self._delayed_tooltip("Defaults to 'animation' if left empty. Saves to documents/Fluoddity/ All filenames get timestamps appended")

        imgui.end()

        if recording_active or video_pending:
            imgui.pop_style_color()
