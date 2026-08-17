"""Audio window: streamline-driven voice settings and pipeline telemetry."""
from imgui_bundle import imgui
from state.audio_state import (AUDIO_BLOCK, AUDIO_SLOTS, AUDIO_RB_BLOCKS,
                               MAX_VOICES, MAX_VOICE_DELAY_SAMPLES)


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

            # === Voices ===
            # MAX_VOICES, not MAX_STREAMLINES: the lane buffer is sized for
            # voices, and the mix reduction is serial over them.
            max_voices = max(1, min(s.count, MAX_VOICES))
            _, a.voice_count = imgui.slider_int(
                "Voice Count", a.voice_count, 1, max_voices,
                flags=imgui.SliderFlags_.logarithmic
            )
            self._delayed_tooltip(
                f"How many particles contribute to the mix (the first N).\n"
                f"Independent of the streamline count: sonifying a whole\n"
                f"swarm is mostly wash, and the mix reduction is serial\n"
                f"over voices, so it falls behind well before the cap of\n"
                f"{MAX_VOICES}."
            )
            if a.voice_count > max_voices:
                imgui.text_disabled(f"  clamped to {max_voices}")

            _, a.rms_normalise = imgui.checkbox(
                "RMS Normalise", a.rms_normalise
            )
            self._delayed_tooltip(
                "Scale the mix by 1/sqrt(voices) so the level holds steady\n"
                "as Voice Count changes. The particles are near-independent\n"
                "(measured correlation ~0.015), so their energy adds as\n"
                "sqrt(n); dividing by n instead would fade toward silence."
            )

            # The ring bounds the delay in SAMPLES, so the millisecond ceiling
            # moves with the sample rate.
            max_delay_ms = MAX_VOICE_DELAY_SAMPLES / max(1, a.sample_rate) * 1000.0

            _, a.voice_delay = imgui.checkbox("Voice Delay", a.voice_delay)
            self._delayed_tooltip(
                "Read each voice a different distance in the past, so they\n"
                "smear against each other instead of landing together.\n"
                "The distance is hashed from the voice index, so it is\n"
                "stable for the life of the stream.\n\n"
                "Expect chorus/smear rather than decorrelation: the voices\n"
                "were already near-independent before this."
            )
            if a.voice_delay:
                _, a.delay_min_ms = imgui.slider_float(
                    "Delay Min", a.delay_min_ms, 0.0, max_delay_ms,
                    format="%.0f ms"
                )
                _, a.delay_max_ms = imgui.slider_float(
                    "Delay Max", a.delay_max_ms, 0.0, max_delay_ms,
                    format="%.0f ms"
                )
                self._delayed_tooltip(
                    f"Delays are drawn from this range. Capped at "
                    f"{max_delay_ms:.0f} ms by the\n"
                    f"depth of the GPU lane ring "
                    f"({MAX_VOICE_DELAY_SAMPLES:,} samples)."
                )
                if a.delay_max_ms < a.delay_min_ms:
                    imgui.text_disabled("  min/max swapped")

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
            imgui.text(f"fences {a.live_fences}/{AUDIO_SLOTS}"
                       f"   overruns {a.overruns}")
            if a.live_fences > AUDIO_SLOTS:
                self._delayed_tooltip(
                    "GL sync objects are leaking. This degrades the display\n"
                    "driver system-wide, not just this app. Please report it."
                )
            elif a.overruns > 0:
                self._delayed_tooltip(
                    "Blocks were produced faster than the device consumed\n"
                    "them and were dropped. Harmless in small numbers."
                )

        imgui.end()
