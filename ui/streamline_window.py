"""Streamlines window: controls for the cursor-seeded streamline probe."""
from imgui_bundle import imgui
from state.streamline_state import MAX_STEPS, MAX_STREAMLINES


class StreamlineWindowMixin:
    """Mixin for streamline probe controls. Combined into UI via multiple inheritance."""

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
                "Releases a test particle at the cursor with no velocity and "
                "traces its path through the canvas force field."
            )
            imgui.separator()

            _, s.steps = imgui.slider_int("Steps", s.steps, 2, MAX_STEPS)
            self._delayed_tooltip(
                "Number of integration steps.\n"
                "Longer paths follow the field further ahead."
            )

            _, s.force_scale = imgui.slider_float(
                "Force Scale", s.force_scale, 0.0, 20.0, format="%.3f"
            )
            self._delayed_tooltip(
                "Converts canvas values into acceleration.\n"
                "Raise this if the streamline barely moves."
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
                "Spring force pulling the particle back toward the seed.\n"
                "0 lets it drift free; higher values tether it to the cursor\n"
                "so it orbits and stays on screen longer.\n"
                "Scaled by Step Size in the shader, so it needs large values\n"
                "to bite - log scale. Lowering Step Size raises the value needed."
            )

            imgui.separator()

            # === Multi-streamline spray ===
            _, s.count = imgui.slider_int("Count", s.count, 1, MAX_STREAMLINES)
            self._delayed_tooltip(
                "Number of streamlines launched from the cursor.\n"
                "Above 1, each gets a random initial velocity so the\n"
                "paths fan out instead of overlapping."
            )

            spray = s.count > 1
            if not spray:
                imgui.begin_disabled()

            _, s.initial_speed = imgui.slider_float(
                "Initial Speed", s.initial_speed, 0.0, 5.0, format="%.3f"
            )
            self._delayed_tooltip(
                "Maximum magnitude of the random launch velocity.\n"
                "Higher values throw the spray wider before the\n"
                "field takes over."
            )

            _, s.resample_each_frame = imgui.checkbox(
                "Resample Each Frame", s.resample_each_frame
            )
            self._delayed_tooltip(
                "Redraw the random launch directions every frame.\n"
                "Uncheck to freeze the fan so only the evolving\n"
                "field moves the paths."
            )

            if not spray:
                imgui.end_disabled()

            imgui.separator()

            changed, color = imgui.color_edit3("Line Color", list(s.color))
            if changed:
                s.color = tuple(color)

            _, s.opacity = imgui.slider_float(
                "Opacity", s.opacity, 0.0, 1.0, format="%.2f"
            )

        imgui.end()
