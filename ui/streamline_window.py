"""Streamlines window: controls for the persistent streamline particles."""
from imgui_bundle import imgui
from state.streamline_state import MAX_STEPS, MAX_STREAMLINES


class StreamlineWindowMixin:
    """Mixin for streamline controls. Combined into UI via multiple inheritance."""

    def render_streamline_window(self):
        """Render the Streamlines control window."""
        expanded, opened = imgui.begin("Streamlines", True)

        if not opened:
            self.state.streamline.enabled = False
            imgui.end()
            return

        if expanded:
            s = self.state.streamline

            imgui.text_wrapped(
                "Read-only Fluoddity particles. They sense and move by the "
                "same rules as real particles but leave no trail, so they "
                "observe the simulation without changing it."
            )

            # === Transport ===
            if imgui.button("Reset to Seed"):
                s.request_reset = True
            self._delayed_tooltip(
                "Re-place every particle at the seed position\n"
                "and clear its path history."
            )
            imgui.same_line()
            _, s.running = imgui.checkbox("Running", s.running)
            self._delayed_tooltip(
                "Pause the tracer without pausing the simulation\n"
                "(or vice versa - the two clocks are independent)."
            )

            # Seed pinning (middle-click in the viewport does the same thing)
            if s.seed_pinned:
                imgui.text(f"Seed pinned at ({s.pinned_seed[0]:.3f}, {s.pinned_seed[1]:.3f})")
                imgui.same_line()
                if imgui.small_button("Unpin"):
                    s.seed_pinned = False
            else:
                imgui.text_disabled("Seed follows cursor")
            self._delayed_tooltip("Middle-click in the viewport to pin or unpin the seed.")

            imgui.separator()

            # === Schedule ===
            imgui.text("Schedule")
            # Audio owns the clock while it is on: one block of samples per
            # dispatch, at the device's sample rate.
            audio_locked = self.state.audio.enabled
            if audio_locked:
                imgui.begin_disabled()

            _, s.dispatch_hz = imgui.slider_float(
                "Dispatch Rate", s.dispatch_hz, 1.0, 2000.0, format="%.0f Hz",
                flags=imgui.SliderFlags_.logarithmic
            )
            if not audio_locked:
                self._delayed_tooltip(
                    "Tracer dispatches per second, independent of frame rate\n"
                    "and of the physics step rate."
                )

            _, s.steps_per_dispatch = imgui.slider_int(
                "Steps / Dispatch", s.steps_per_dispatch, 1, 512
            )
            if not audio_locked:
                self._delayed_tooltip(
                    "Integration steps advanced per dispatch.\n"
                    "Rate x Steps = integration steps per second."
                )

            if audio_locked:
                imgui.end_disabled()

            imgui.text_disabled(
                f"  = {s.dispatch_hz * s.steps_per_dispatch:,.0f} steps/sec"
            )
            if audio_locked:
                imgui.text_disabled("  locked by Audio")

            _, s.field_interpolation = imgui.checkbox(
                "Field Interpolation", s.field_interpolation
            )
            self._delayed_tooltip(
                "Blend the sensed field between the previous and current\n"
                "canvas frame instead of sampling it as a staircase.\n"
                "\n"
                "The canvas only updates once per physics step, which is\n"
                "dozens of audio samples apart, so without this the field is\n"
                "piecewise constant. Turn it off to hear whether a buzz at\n"
                "the physics rate comes from the interpolation or from the\n"
                "field genuinely stepping."
            )
            if not s.field_interpolation:
                imgui.text_disabled("  field steps at the physics rate")

            imgui.separator()

            # === Integration ===
            imgui.text("Physics")
            _, s.newtonian_mode = imgui.checkbox(
                "Newtonian Mode", s.newtonian_mode
            )
            self._delayed_tooltip(
                "Trace the canvas as a force field instead of running the\n"
                "full Fluoddity particle.\n\n"
                "Off (default): streamers are read-only Fluoddity particles.\n"
                "They read two sensors and evaluate the same rule real\n"
                "particles do, but leave no trail, so they show what the\n"
                "simulation's own particles would do.\n\n"
                "On: the original tracer. The canvas value is applied\n"
                "directly as acceleration, so the particle advects along the\n"
                "flow. Simpler, and it traces the field rather than\n"
                "reproducing behaviour - a different look and sound, not a\n"
                "more accurate one.")

            if s.newtonian_mode:
                _, s.force_scale = imgui.slider_float(
                    "Force Scale", s.force_scale, 0.0, 5.0, format="%.3f"
                )
                self._delayed_tooltip(
                    "Canvas value to acceleration. Higher follows the field\n"
                    "more sharply.")

                _, s.damping = imgui.slider_float(
                    "Damping", s.damping, 0.5, 1.0, format="%.5f"
                )
                self._delayed_tooltip(
                    "Velocity retained per step. 1.0 is frictionless, so the\n"
                    "particle keeps its momentum and overshoots the flow.\n"
                    "Useful values sit very close to 1.")

                # Lower bound is well below the saved values from when this
                # was the only mode (0.0004 is typical), so loading an old
                # preferences file does not silently clamp them.
                _, s.step_size = imgui.slider_float(
                    "Step Size", s.step_size, 0.0001, 0.1, format="%.5f"
                )
                self._delayed_tooltip(
                    "Velocity to displacement per integration step.")

                _, s.restore_force = imgui.slider_float(
                    "Restore Force", s.restore_force, 0.0, 2000.0, format="%.0f"
                )
                self._delayed_tooltip(
                    "Spring pulling each particle back toward its own offset\n"
                    "point near the seed. 0 lets them drift free. Scaled by\n"
                    "Step Size internally and clamped for stability, so the\n"
                    "useful range is large.")
            else:
                imgui.text_wrapped(
                    "Streamers are read-only Fluoddity particles: they "
                    "evaluate the same rules and physics sliders real "
                    "particles do, but leave no trail. Their behaviour comes "
                    "from the Physics window, not from here.")

            imgui.separator()

            # === Population ===
            imgui.text("Population")
            _, s.count = imgui.slider_int("Count", s.count, 1, MAX_STREAMLINES)
            self._delayed_tooltip(
                "Number of particles. Changing this re-seeds them all."
            )

            _, s.initial_speed = imgui.slider_float(
                "Initial Speed", s.initial_speed, 0.0, 5.0, format="%.3f"
            )
            self._delayed_tooltip(
                "Maximum magnitude of the random launch velocity\n"
                "given to each particle when it is seeded."
            )

            _, s.seed_scatter = imgui.slider_float(
                "Seed Scatter", s.seed_scatter, 0.0, 1.0, format="%.3f"
            )
            self._delayed_tooltip(
                "Radius of the disc each particle spawns into around the\n"
                "seed. At 0 they all start from the same point and, running\n"
                "identical rules, trace near-identical paths."
            )

            _, s.resample_each_frame = imgui.checkbox(
                "Resample Launch Directions", s.resample_each_frame
            )
            self._delayed_tooltip(
                "Draw fresh random launch directions on each reset\n"
                "and respawn. Uncheck for a repeatable fan."
            )

            _, s.hazard_rate = imgui.slider_float(
                "Hazard Rate", s.hazard_rate, 0.0, 0.1, format="%.5f",
                flags=imgui.SliderFlags_.logarithmic
            )
            self._delayed_tooltip(
                "Chance per integration step that a particle respawns\n"
                "at the seed, giving the population a constant turnover.\n"
                "Per step, not per frame, so the rate is unaffected by\n"
                "Dispatch Rate and Steps / Dispatch."
            )
            if s.hazard_rate > 0.0:
                # Mean lifetime of a geometric distribution is 1/p.
                steps_per_sec = max(1.0, s.dispatch_hz * s.steps_per_dispatch)
                imgui.text_disabled(
                    f"  mean life {1.0 / s.hazard_rate:,.0f} steps"
                    f" ({1.0 / s.hazard_rate / steps_per_sec:.2f} s)"
                )

            _, s.stop_at_edge = imgui.checkbox("Stop At Edge", s.stop_at_edge)
            self._delayed_tooltip(
                "Retire particles that leave the canvas.\n"
                "Unchecked, they slide along the boundary instead."
            )

            if not s.stop_at_edge:
                imgui.begin_disabled()
            _, s.respawn_at_seed = imgui.checkbox("Respawn At Seed", s.respawn_at_seed)
            self._delayed_tooltip(
                "Retired particles restart at the seed, so the\n"
                "population keeps streaming indefinitely."
            )
            if not s.stop_at_edge:
                imgui.end_disabled()

            imgui.separator()

            # === Display ===
            imgui.text("Display")
            _, s.tail_length = imgui.slider_int(
                "Tail Length", s.tail_length, 2, MAX_STEPS
            )
            self._delayed_tooltip(
                f"Ring entries drawn per particle (max {MAX_STEPS:,}).\n"
                "Older history is overwritten as the ring wraps."
            )

            changed, color = imgui.color_edit3("Line Color", list(s.color))
            if changed:
                s.color = tuple(color)

            _, s.opacity = imgui.slider_float(
                "Opacity", s.opacity, 0.0, 1.0, format="%.2f"
            )

            # Cap the slider at what this driver will actually rasterize.
            svc = getattr(self, 'streamline_service', None)
            max_width = svc.max_line_width if svc is not None else 1.0
            if max_width > 1.0:
                _, s.line_width = imgui.slider_float(
                    "Line Width", s.line_width, 1.0, max_width, format="%.1f"
                )
                self._delayed_tooltip(
                    "Line thickness in pixels (glLineWidth).\n"
                    f"This driver supports up to {max_width:.0f}px."
                )
            else:
                imgui.text_disabled("Line Width: 1px (driver max)")
                self._delayed_tooltip(
                    "This driver only rasterizes 1px lines in core profile."
                )

        imgui.end()
