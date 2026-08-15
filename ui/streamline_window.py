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
                "Persistent particles carried by the canvas force field. "
                "They keep streaming until reset."
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
            _, s.dispatch_hz = imgui.slider_float(
                "Dispatch Rate", s.dispatch_hz, 1.0, 2000.0, format="%.0f Hz",
                flags=imgui.SliderFlags_.logarithmic
            )
            self._delayed_tooltip(
                "Tracer dispatches per second, independent of frame rate\n"
                "and of the physics step rate."
            )

            _, s.steps_per_dispatch = imgui.slider_int(
                "Steps / Dispatch", s.steps_per_dispatch, 1, 512
            )
            self._delayed_tooltip(
                "Integration steps advanced per dispatch.\n"
                "Rate x Steps = integration steps per second."
            )
            imgui.text_disabled(
                f"  = {s.dispatch_hz * s.steps_per_dispatch:,.0f} steps/sec"
            )

            imgui.separator()

            # === Integration ===
            imgui.text("Integration")
            _, s.force_scale = imgui.slider_float(
                "Force Scale", s.force_scale, 0.0, 20.0, format="%.3f"
            )
            self._delayed_tooltip(
                "Converts canvas values into acceleration.\n"
                "Raise this if the particles barely move."
            )

            _, s.damping = imgui.slider_float(
                "Damping", s.damping, 0.5, 1.0, format="%.4f"
            )
            self._delayed_tooltip(
                "Velocity retained each step.\n"
                "1.0 is frictionless; lower values settle the path\n"
                "into the field direction more tightly."
            )

            _, s.step_size = imgui.slider_float(
                "Step Size", s.step_size, 0.0001, 0.1, format="%.4f"
            )
            self._delayed_tooltip(
                "Distance travelled per step.\n"
                "Lower values trace a smoother, shorter path."
            )

            _, s.restore_force = imgui.slider_float(
                "Restore Force", s.restore_force, 0.0, 10000.0, format="%.1f",
                flags=imgui.SliderFlags_.logarithmic
            )
            self._delayed_tooltip(
                "Spring force pulling particles back toward the seed.\n"
                "Raise it and the population gathers at the cursor, so you\n"
                "can drag the swarm around the canvas.\n"
                "Scaled by Step Size in the shader, so it needs large values\n"
                "to bite - log scale. Lowering Step Size raises what you need."
            )

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

            _, s.resample_each_frame = imgui.checkbox(
                "Resample Launch Directions", s.resample_each_frame
            )
            self._delayed_tooltip(
                "Draw fresh random launch directions on each reset\n"
                "and respawn. Uncheck for a repeatable fan."
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
