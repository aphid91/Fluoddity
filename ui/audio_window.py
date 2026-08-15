"""Audio window: streamline-driven voice settings and pipeline telemetry."""
from imgui_bundle import imgui
from state.audio_state import AUDIO_BLOCK, AUDIO_SLOTS, AUDIO_RB_BLOCKS
from state.streamline_state import MAX_STREAMLINES


class AudioWindowMixin:
    """Mixin for audio controls. Combined into UI via multiple inheritance."""

    def render_audio_window(self):
        """Render the Audio control window."""
        expanded, opened = imgui.begin("Audio", True)

        if not opened:
            self.state.audio.enabled = False
            imgui.end()
            return

        if expanded:
            a = self.state.audio
            s = self.state.streamline

            imgui.text_wrapped(
                "One voice driven by a single streamline. The sample at each "
                "integration step is dot(velocity, field)."
            )

            if a.last_error:
                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
                                   a.last_error)

            imgui.separator()

            # === Voice ===
            max_voice = max(0, min(s.count, MAX_STREAMLINES) - 1)
            _, a.voice_index = imgui.slider_int(
                "Voice Particle", a.voice_index, 0, max(0, max_voice)
            )
            self._delayed_tooltip(
                "Which streamline drives the voice.\n"
                "Combining several particles comes later."
            )

            _, a.amplitude = imgui.slider_float(
                "Amplitude", a.amplitude, 0.0, 4.0, format="%.3f"
            )
            self._delayed_tooltip("Gain applied to dot(velocity, field).")

            _, a.auto_gain = imgui.checkbox("Auto Gain", a.auto_gain)
            self._delayed_tooltip(
                "Normalise by a tracked peak before applying Amplitude.\n"
                "The raw dot product scales with canvas energy and spans\n"
                "orders of magnitude, so without this Amplitude would need\n"
                "to be retuned constantly."
            )
            if a.auto_gain:
                imgui.text_disabled(f"  tracking {a.auto_gain_db:+.1f} dB")

            _, a.highpass_hz = imgui.slider_float(
                "High Pass", a.highpass_hz, 1.0, 500.0, format="%.0f Hz",
                flags=imgui.SliderFlags_.logarithmic
            )
            self._delayed_tooltip(
                "One-pole DC blocker. The raw dot product has a large DC\n"
                "component, which would otherwise offset the whole signal."
            )

            _, a.reset_ramp_ms = imgui.slider_float(
                "Reset Ramp", a.reset_ramp_ms, 0.0, 50.0, format="%.1f ms"
            )
            self._delayed_tooltip(
                "Gain ramp applied after a particle resets. The high pass\n"
                "removes the DC step a reset causes, but the instantaneous\n"
                "jump in dot() still clicks without this."
            )

            _, a.limiter_ceiling = imgui.slider_float(
                "Limiter Ceiling", a.limiter_ceiling, 0.1, 1.0, format="%.2f"
            )

            imgui.separator()

            # === Telemetry ===
            imgui.text("Pipeline")
            if a.device_name:
                imgui.text_disabled(f"  {a.device_name}")
            latency_ms = ((AUDIO_BLOCK * AUDIO_RB_BLOCKS
                           + a.in_flight * AUDIO_BLOCK)
                          / max(1, a.sample_rate) * 1000.0)
            imgui.text(f"latency ~{latency_ms:.0f} ms"
                       f"   in-flight {a.in_flight}/{AUDIO_SLOTS}")
            imgui.progress_bar(a.ring_fill, imgui.ImVec2(-1, 0),
                               f"ring {a.ring_fill * 100:.0f}%")
            imgui.text(f"peak {a.peak:.3f}   starves {a.starves}")
            if a.starves > 0:
                self._delayed_tooltip(
                    "The audio ring ran dry. Raise the block count or\n"
                    "reduce per-frame work if this climbs steadily."
                )

        imgui.end()
