"""Render settings window: unified per-renderer controls.

One window that shows the active renderer's settings (Preferences -> Renderer):
- OpenGL: the volumetric path tracer (medium / lighting / sky / post-process).
- Optix:  the OptiX path tracer (geometry / material / lighting / sky /
          rasterize / path-trace / post-process).

The first two sections (RT mode + capture, then Camera) are shared and use the
OptiX Controls labeling/layout. Merged from the former 3D Controls, OptiX
Controls, and Tracer windows in the 3D-only cleanup.
"""
from imgui_bundle import imgui
from camera_input import sync_orbit_angles_from_camera


class RenderSettingsWindowMixin:
    """Mixin for the unified Render settings window."""

    # Preview request flags (set by UI, cleared by orchestrator). One per backend
    # so the orchestrator can start each render with the real framebuffer size.
    _request_optix_preview: bool = False
    _request_tracer_preview: bool = False

    def _persisted_header(self, label, pref_field):
        """Collapsing header whose open/closed state persists in preferences.

        Drives the header from ``ui_windows.<pref_field>`` each frame (making the
        preference authoritative over imgui's .ini) and writes the state back on
        user toggle. Mirrors the physics-window group pattern. Returns the open bool.
        """
        uw = self.state.preferences.ui_windows
        imgui.set_next_item_open(getattr(uw, pref_field))
        is_open = imgui.collapsing_header(label)
        if imgui.is_item_toggled_open():
            setattr(uw, pref_field, is_open)
        return is_open

    def render_render_settings_window(self):
        """Render the unified Render settings window for the active renderer."""
        visible, opened = imgui.begin("Render settings", True)
        if not opened:
            self.state.preferences.ui_windows.show_render_settings_window = False
            imgui.end()
            return
        if not visible:
            imgui.end()
            return

        # Renderer selection at the very top, then shared appearance controls.
        self._render_renderer_dropdown()
        self._render_appearance_top()
        imgui.separator()

        if self.state.preferences.rendering.renderer == 1:
            self._render_optix_settings()
        else:
            self._render_opengl_settings()

        imgui.end()

    # ------------------------------------------------------------------ shared
    def _render_renderer_dropdown(self):
        """Renderer selection (OpenGL vs Optix) — drives the whole app."""
        from pathtracer_interface import PathTracerInterface
        r = self.state.preferences.rendering
        renderer_labels = ["OpenGL", "Optix"]
        optix_available = PathTracerInterface.is_available()
        cur = r.renderer if 0 <= r.renderer < len(renderer_labels) else 0
        clicked, new_renderer = imgui.combo("Renderer", cur, renderer_labels)
        if clicked and not (new_renderer == 1 and not optix_available):
            r.renderer = new_renderer
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "OpenGL: GL points (Pathtrace Off) or the volumetric path tracer.\n"
                "Optix: the RTX path tracer (rasterize / path-trace modes).\n"
                "Requires an NVIDIA RTX GPU with OptiX/CUDA installed for Optix.")

    def _render_appearance_top(self):
        """Brightness + Tonemap Softness — shown at the top for both renderers."""
        r = self.state.preferences.rendering
        _, r.brightness = imgui.slider_float(
            "Brightness", r.brightness, 0.01, 2.5, format="%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Global brightness multiplier for the output.")
        _, r.tonemap_softness = imgui.slider_float(
            "Tonemap Softness", r.tonemap_softness, 0.1, 4.0, format="%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls highlight compression (asinh stretch).\n"
                "Low = more linear (brighter highlights).\n"
                "High = more logarithmic (reveals faint detail).")

    def _render_bloom_controls(self):
        """Bloom checkbox + sliders — shared, shown inside Post-Process."""
        b = self.state.preferences.bloom
        _, b.enabled = imgui.checkbox("Bloom", b.enabled)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a glow effect around bright areas.")
        if b.enabled:
            imgui.indent(20)
            _, b.threshold = imgui.slider_float("Threshold", b.threshold, 0.0, 2.0, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Brightness cutoff for bloom extraction.\nLower = more glow everywhere.")
            _, b.intensity = imgui.slider_float("Bloom intensity", b.intensity, 0.0, 3.0, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Strength of the bloom glow.")
            _, b.radius = imgui.slider_float("Radius", b.radius, 0.1, 3.0, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Spread of the bloom blur kernel.")
            imgui.unindent(20)

    def _render_pathtrace_mode_row(self, *, show_spp_slider):
        """Pathtrace mode cycling button + shared samples slider.

        The button cycles the shared ``rendering.rt_mode`` (0=Off, 1=X spp,
        2=Accumulate) with unified "Pathtrace:" labels for both renderers.
        The X-spp samples slider (``rendering.rt_samples``) is only shown when
        ``show_spp_slider`` (OptiX) — the volumetric tracer is always 1 spp in
        realtime mode. In OptiX Off (rasterize) mode the OptiX-only rasterize
        samples slider is shown instead.
        """
        r = self.state.preferences.rendering
        labels = ["Pathtrace: Off", f"Pathtrace: {r.rt_samples} spp", "Pathtrace: Accumulate"]
        if imgui.button(labels[r.rt_mode]):
            r.rt_mode = (r.rt_mode + 1) % 3
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Toggle between render modes. Off is fastest. In SPP mode, each\n"
                "frame will be slightly noisy (if using Optix, try the denoiser!).\n"
                "In Accumulate mode, frames will all blend together, reducing\n"
                "noise (best when simulation is paused). Moving the camera resets\n"
                "the accumulation buffer.")
        if show_spp_slider and r.rt_mode == 1:
            imgui.same_line()
            imgui.set_next_item_width(100)
            _, r.rt_samples = imgui.slider_int("##rt_samples", r.rt_samples, 1, 8)
        elif show_spp_slider and r.rt_mode == 0:
            # OptiX rasterize (Off) mode has its own samples-per-frame count.
            imgui.same_line()
            imgui.set_next_item_width(100)
            _, self.state.preferences.optix.rz_samples = imgui.slider_int(
                "##rz_samples", self.state.preferences.optix.rz_samples, 1, 8)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Rasterize samples per frame (averaged before denoise)")

    def _render_resolution_scale_row(self):
        """Resolution Scale (shared ``rendering.render_resolution_scale``)."""
        r = self.state.preferences.rendering
        imgui.set_next_item_width(100)
        changed, new_scale = imgui.input_float(
            "Resolution Scale", r.render_resolution_scale, 0.0, 0.0, "%.2f")
        if imgui.is_item_deactivated_after_edit():
            r.render_resolution_scale = max(0.1, min(4.0, new_scale))

    def _render_firefly_clamp(self):
        """Firefly Clamp checkbox + max (shared ``rendering.firefly_clamp``)."""
        r = self.state.preferences.rendering
        _, r.firefly_clamp = imgui.checkbox("Firefly Clamp", r.firefly_clamp)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Limits the radiance of any individual sampled pixel, useful\n"
                "when scenes remain noisy after many samples.")
        if r.firefly_clamp:
            imgui.set_next_item_width(imgui.get_content_region_avail().x)
            _, r.firefly_clamp_max = imgui.drag_float(
                "##firefly_max", r.firefly_clamp_max, 0.1, 0.1, 1000.0, "Max: %.1f")

    def _render_camera_section(self):
        """Camera section — identical for both renderers (shared camera state)."""
        if self._persisted_header("Camera", "render_group_camera"):
            cam = self.state.camera
            _, cam.fov = imgui.slider_float(
                "FOV", cam.fov, 10.0, 95.0, format="%.0f deg")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Viewing angle in degrees. Strong depth of field tends to\n"
                    "look better at low values (25ish).")
            _, cam.aperture = imgui.slider_float(
                "Aperture", cam.aperture, 0.0, 0.15, format="%.3f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Controls the strength of the depth-of-field effect. Adjust\n"
                    "the focal depth until the target is in focus, or hover over\n"
                    "it and press the focus key.")
            _, cam.focal_plane_depth = imgui.slider_float(
                "Focal Depth", cam.focal_plane_depth, 0.1, 10.0, format="%.1f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Controls the distance at which objects are in focus. Adjust\n"
                    "until the target is in focus, or hover over it and press the\n"
                    "focus key.")
            _, cam.move_speed = imgui.slider_float(
                "Controller Move Speed", cam.move_speed, 0.1, 5.0, format="%.1f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("How fast does the controller move the camera?")
            _, cam.rotate_speed = imgui.slider_float(
                "Rotate Speed", cam.rotate_speed, 0.1, 5.0, format="%.1f")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "How fast does the camera orbit / how sensitive is the\n"
                    "controller Look?")
            changed, values = imgui.drag_float3(
                "Orbit Center", list(cam.orbit_center), 0.01, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Where does the camera orbit when moved with WASD?")
            if changed:
                cam.orbit_center[0], cam.orbit_center[1], cam.orbit_center[2] = values
                if self.tracer_controller_cam is not None:
                    sync_orbit_angles_from_camera(cam, self.tracer_controller_cam)
            _, cam.orbit_rate = imgui.slider_float(
                "Orbit Rate", cam.orbit_rate, -0.02, 0.02, format="%.4f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Automatic rotation for lazy-Susan video recording.")

            imgui.separator()
            _, cam.stereogram = imgui.checkbox("Stereogram", cam.stereogram)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Enable Stereo mode for full 3D cross-eye viewing.")
            if cam.stereogram:
                _, cam.eye_offset = imgui.slider_float(
                    "Eye Offset", cam.eye_offset, 0.0, 0.5, format="%.3f")
                # Parallel vs Toe-in convergence toggle
                if imgui.radio_button("Parallel", not cam.stereo_toe_in):
                    cam.stereo_toe_in = False
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Both eyes look straight ahead.")
                imgui.same_line()
                if imgui.radio_button("Toe-in", cam.stereo_toe_in):
                    cam.stereo_toe_in = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Eyes converge on the Focal Depth plane.")
                mode_label = "Wall-eye" if cam.stereo_wall_eye else "Cross-eye"
                if imgui.button(f"Current mode: {mode_label}"):
                    cam.stereo_wall_eye = not cam.stereo_wall_eye
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Toggle which eye renders on which half of the "
                        "screen. Some people find wall-eye viewing easier.")

    def _render_lighting_section(self):
        """Lighting section — shared LightingPrefs, identical for both renderers.

        (Photosphere + Enable NEE live here; the OptiX-only Cos-lobe Sky toggle
        is added by the OptiX path.)
        """
        if self._persisted_header("Lighting", "render_group_lighting"):
            lit = self.state.preferences.lighting
            changed, vals = imgui.drag_float3(
                "Light Dir", list(lit.light_direction), 0.01, -1.0, 1.0)
            if changed:
                lit.light_direction = list(vals)
            _, lit.light_color = imgui.color_edit3(
                "Light Color", lit.light_color)
            _, lit.light_intensity = imgui.slider_float(
                "Intensity", lit.light_intensity, 0.0, 20.0)

            _, lit.nee = imgui.checkbox("Enable NEE", lit.nee)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Next Event Estimation: trace a shadow ray toward the\n"
                    "light for direct lighting. In rasterize mode, turning\n"
                    "this off leaves only the ambient + AO term.")
            # OptiX-only cos-lobe sky toggle (added by caller when in Optix mode)
            if self.state.preferences.rendering.renderer == 1:
                _, self.state.preferences.optix.pt_env_sky_nee = imgui.checkbox(
                    "Cos-lobe Sky", self.state.preferences.optix.pt_env_sky_nee)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Replace legacy directional sun + gradient sky\n"
                        "with a cosine-lobe environment model.\n"
                        "Sky color controls hemisphere glow,\n"
                        "sun direction/color/intensity control sun disk.")
                if self.state.preferences.optix.pt_env_sky_nee:
                    _, self.state.preferences.optix.pt_sun_exp = imgui.slider_float(
                        "Sun Sharpness", self.state.preferences.optix.pt_sun_exp,
                        1.0, 256.0, format="%.0f")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(
                            "Exponent of the sun's cosine-power lobe.\n"
                            "Higher = tighter, sharper sun disk.")
            changed_photo, lit.photosphere = imgui.checkbox(
                "Photosphere", lit.photosphere)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Replaces the skybox with a photosphere loaded from:\n"
                    "volrender/textures/skybox.jpg")
            # On the OpenGL side the photosphere needs its skybox texture loaded.
            if (changed_photo and lit.photosphere
                    and self.state.preferences.rendering.renderer == 0):
                ti = self._tracer_interface
                if ti is not None:
                    if ti._skybox_tex is None:
                        ti._skybox_tex = ti._load_skybox()
                    if ti._skybox_tex is None:
                        lit.photosphere = False

    def _render_sky_section(self):
        """Sky section — two-tone gradient from the shared LightingPrefs."""
        if self._persisted_header("Sky", "render_group_sky"):
            lit = self.state.preferences.lighting
            _, lit.sky_color_top = imgui.color_edit3("Sky Top", lit.sky_color_top)
            _, lit.sky_color_bottom = imgui.color_edit3("Sky Bottom", lit.sky_color_bottom)
            if self.state.preferences.rendering.renderer == 0:
                # Sky intensity applies to the volumetric tracer's gradient.
                _, lit.sky_intensity = imgui.slider_float(
                    "Sky Intensity", lit.sky_intensity, 0.0, 1.0)

    # ------------------------------------------------------------------- OptiX
    def _render_optix_settings(self):
        p = self.state.preferences

        # ---- Pathtrace mode button + unlabeled samples slider (shared) ----
        # Samples slider (spp) shows only in "X spp" mode and only for OptiX.
        self._render_pathtrace_mode_row(show_spp_slider=True)

        # ---- Capture SPP + Re-render Preview (shared) ----
        r = p.rendering
        _, r.capture_spp = imgui.slider_int("Capture SPP", r.capture_spp, 1, 128)
        rt_active = r.rt_mode > 0
        if rt_active:
            imgui.begin_disabled()
        if imgui.button("Re-render pathtrace preview"):
            self._request_optix_preview = True
        if rt_active:
            imgui.end_disabled()

        # Progress / status indicator
        pt_interface = getattr(self, '_pathtracer_interface', None)
        if r.rt_mode == 1:
            imgui.same_line()
            imgui.text(f"  [{r.rt_samples} spp]")
        elif r.rt_mode == 2:
            imgui.same_line()
            count = self.state.camera.pathtracer_sample_count
            imgui.text(f"  [accum: {count} spp]")
        elif pt_interface is not None and pt_interface.preview_active:
            imgui.same_line()
            done = pt_interface.preview_samples_done
            target = pt_interface.preview_target_spp
            imgui.text(f"  [{done}/{target} spp]")
        elif pt_interface is not None and pt_interface.preview_has_result:
            imgui.same_line()
            imgui.text(f"  [done: {pt_interface.preview_last_spp} spp]")

        # ---- Resolution Scale (applied on Enter key, shared) ----
        self._render_resolution_scale_row()

        imgui.separator()

        rasterize = (r.rt_mode == 0)

        # ---- Camera (shared) ----
        self._render_camera_section()

        # ---- Geometry ----
        if self._persisted_header("Geometry", "render_group_geometry"):
            _, p.optix.sphere_radius_scale = imgui.slider_float(
                "Particle Size", p.optix.sphere_radius_scale,
                0.1, 10.0, format="%.1fx")
            if not p.optix.use_curves:
                _, p.optix.sphere_size_jitter = imgui.slider_float(
                    "Sphere Jitter", p.optix.sphere_size_jitter,
                    0.0, 1.0, format="%.2f")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Per-sphere radius jitter to reduce banding artifacts")

            _, p.optix.use_curves = imgui.checkbox("Curves", p.optix.use_curves)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Render entities as round linear curves oriented along velocity")
            if p.optix.use_curves:
                _, p.optix.curve_length = imgui.slider_float(
                    "Curve Length", p.optix.curve_length, 0.0, 10.0, format="%.2f")
                _, p.optix.curve_r0 = imgui.slider_float(
                    "Tip Radius", p.optix.curve_r0, 0.01, 5.0, format="%.2f")
                _, p.optix.curve_r1 = imgui.slider_float(
                    "Base Radius", p.optix.curve_r1, 0.01, 5.0, format="%.2f")

            _, p.optix.sdf_enabled = imgui.checkbox("Enable Dish", p.optix.sdf_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Adds a curved plate near the ground plane with which\n"
                    "particles can collide.")

        # ---- Material (moved to between Geometry and Lighting) ----
        if self._persisted_header("Material", "render_group_material"):
            mat_labels = ["Lambert", "Glossy", "Mirror"]
            _, p.optix.pt_global_material = imgui.combo(
                "BRDF", p.optix.pt_global_material, mat_labels)
            if p.optix.pt_global_material == 1:  # Glossy
                _, p.optix.pt_glossy_ior = imgui.slider_float(
                    "Glossy IOR", p.optix.pt_glossy_ior, 1.0, 3.0, format="%.2f")
            _, p.optix.albedo_saturation = imgui.slider_float(
                "Albedo Saturation", p.optix.albedo_saturation, 0.0, 1.0)
            _, p.optix.albedo_brightness = imgui.slider_float(
                "Albedo Brightness", p.optix.albedo_brightness, 0.0, 1.0)

        # ---- Lighting (shared) + Sky (shared) ----
        self._render_lighting_section()
        self._render_sky_section()

        # ---- Rasterize-specific (greyed out in path-trace modes) ----
        if self._persisted_header("Rasterize", "render_group_rasterize"):
            if not rasterize:
                imgui.begin_disabled()
            # (The "Shadows (rasterize)" checkbox was removed — always on.)
            _, p.optix.ambient = imgui.slider_float(
                "Ambient (rasterize)", p.optix.ambient, 0.0, 1.0)
            _, p.optix.ambient_color = imgui.color_edit3(
                "Ambient Color", p.optix.ambient_color)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Ambient tint (scaled by Ambient), modulated by AO")

            _, p.optix.rz_depth_of_field = imgui.checkbox(
                "Depth of Field", p.optix.rz_depth_of_field)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Use the thin-lens camera (Aperture / Focal Depth in the\n"
                    "Camera section) in rasterize mode. Nearly free with the denoiser on.")

            _, p.optix.ao_enabled = imgui.checkbox(
                "Ambient Occlusion (rasterize)", p.optix.ao_enabled)
            if p.optix.ao_enabled:
                _, p.optix.ao_num_rays = imgui.slider_int(
                    "AO Rays", p.optix.ao_num_rays, 1, 16)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Determines the number of short ambient occlusion rays to launch.")
                _, p.optix.ao_radius = imgui.slider_float(
                    "AO Radius", p.optix.ao_radius, 0.01, 5.0, format="%.2f")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Determines the range over which ambient occlusion is estimated.")
            if not rasterize:
                imgui.end_disabled()

        # ---- Path trace-specific (greyed out in rasterize mode) ----
        if self._persisted_header("Path Trace", "render_group_pathtrace"):
            if rasterize:
                imgui.begin_disabled()
            _, p.optix.pt_max_bounces = imgui.drag_int(
                "Max Bounces", p.optix.pt_max_bounces, 0.1, 0, 64)
            if imgui.is_item_hovered():
                imgui.set_tooltip("0 = unbounded (Russian roulette only)")
            _, p.optix.pt_rr_start_depth = imgui.slider_int(
                "RR Start Depth", p.optix.pt_rr_start_depth, 1, 16)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "How many bounces are guaranteed before russian roulette begins?")
            # _, p.optix.pt_emission_intensity = imgui.slider_float(
            #     "Emission Intensity", p.optix.pt_emission_intensity,
            #     0.0, 100.0, format="%.1f")
            # if imgui.is_item_hovered():
            #     imgui.set_tooltip("Radiance multiplier for emissive entities (negative hue)")
            if rasterize:
                imgui.end_disabled()

        # ---- Post-Process ----
        if self._persisted_header("Post-Process", "render_group_postprocess"):
            self._render_firefly_clamp()

            _, p.optix.pt_denoise_enabled = imgui.checkbox(
                "Denoise (path trace)", p.optix.pt_denoise_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Apply Optix AI denoiser to finished frames.\n"
                    "(Can be a little expensive at high resolution.)")
            _, p.optix.rz_denoise_enabled = imgui.checkbox(
                "Denoise (rasterize)", p.optix.rz_denoise_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Apply Optix AI denoiser to finished frames.\n"
                    "(Can be a little expensive at high resolution.)")

            self._render_bloom_controls()

        # ---- Timing display ----
        gas_ms = self.state.camera.pathtracer_gas_time_ms
        render_ms = self.state.camera.pathtracer_render_time_ms
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
            f"GAS {gas_ms:.1f}ms  Render {render_ms:.1f}ms")

    # ------------------------------------------------------------------ OpenGL
    def _render_opengl_settings(self):
        # Lazy-create the TracerInterface (owns the volumetric renderer).
        if self._tracer_interface is None:
            from tracer_interface import TracerInterface
            self._tracer_interface = TracerInterface(self.ctx)
            self._apply_tracer_preferences(self._tracer_interface)
        ti = self._tracer_interface

        r = self.state.preferences.rendering

        # The progressive "Re-render Preview" render is now ticked in the main
        # loop (App.run) and displayed fullscreen in the Viewer, matching the
        # OptiX preview — no per-window tick or inline image here.

        # ---- Pathtrace mode button (shared; no per-frame spp slider here) ----
        # The volumetric tracer has no per-frame sample count, so the OptiX
        # X-spp samples slider is hidden for OpenGL (show_spp_slider=False).
        self._render_pathtrace_mode_row(show_spp_slider=False)

        # ---- Capture SPP + Re-render (shared) ----
        _, r.capture_spp = imgui.slider_int("Capture SPP", r.capture_spp, 1, 512)
        rt_active = r.rt_mode > 0
        if rt_active:
            imgui.begin_disabled()
        if imgui.button("Re-render pathtrace preview"):
            self._request_tracer_preview = True
        if rt_active:
            imgui.end_disabled()

        # Progress / status indicator
        if r.rt_mode == 1:
            imgui.same_line()
            imgui.text("  [1spp]")
        elif r.rt_mode == 2:
            imgui.same_line()
            imgui.text(f"  [accum: {ti.samples_done} spp]")
        elif ti.is_rendering:
            imgui.same_line()
            imgui.text(f"  [{ti.samples_done}/{r.capture_spp} spp]")
        elif ti.has_result:
            imgui.same_line()
            imgui.text(f"  [done: {ti.last_spp} spp]")

        # ---- Resolution Scale (applied on Enter key, shared) ----
        self._render_resolution_scale_row()

        imgui.separator()

        # ---- Camera (shared) ----
        self._render_camera_section()

        # ---- Path Trace (Enable SDF moved to the bottom of this section) ----
        if self._persisted_header("Path Trace", "render_group_medium"):
            # _, ti.colored_extinction = imgui.checkbox(
            #     "Colored Extinction", ti.colored_extinction)
            if ti.colored_extinction:
                _, ti.extinction_rgb = imgui.color_edit3("Albedo RGB", ti.extinction_rgb)
                _, ti.albedo_saturation = imgui.slider_float(
                    "Ext Saturation", ti.albedo_saturation, 0.0, 1.0)
                _, ti.albedo_brightness = imgui.slider_float(
                    "Ext Brightness", ti.albedo_brightness, 0.0, 1.0)
            else:
                _, ti.extinction_rgb = imgui.color_edit3("Extinction RGB", ti.extinction_rgb)
                _, ti.albedo_saturation = imgui.slider_float(
                    "Albedo Saturation", ti.albedo_saturation, 0.0, 1.0)
                _, ti.albedo_brightness = imgui.slider_float(
                    "Albedo Brightness", ti.albedo_brightness, 0.0, 1.0)
            #_, ti.density_scale = imgui.drag_float(
            #    "Density Scale", ti.density_scale, 0.00001, 0.00001, 10.0, "%.5f")
            _, ti.density_scale = imgui.slider_float(
                "Density Scale", ti.density_scale, 0.000001, 0.0005,format="%.7f")
            _, ti.hg_g = imgui.slider_float("Scattering (g)", ti.hg_g, -1.0, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("HG phase: -1 back, 0 isotropic, +1 forward")
            # _, ti.emission_strength = imgui.drag_float(
            #     "Emission", ti.emission_strength, 0.01, 0.0, 100.0, "%.3f")
            # if imgui.is_item_hovered():
            #     imgui.set_tooltip("Self-emission intensity (0 = off)")
            _, ti.max_bounces = imgui.drag_int("Max Bounces", ti.max_bounces, 0.1, 0, 64)
            if imgui.is_item_hovered():
                imgui.set_tooltip("0 = unbounded (Russian roulette only)")
            # Enable Dish at the bottom of Medium (was a separate "SDF Scene" header)
            _, ti.sdf_enabled = imgui.checkbox("Enable Dish", ti.sdf_enabled)

        # ---- Lighting (shared) + Sky (shared) ----
        self._render_lighting_section()
        self._render_sky_section()

        # ---- Grid Resolutions (between Sky and Post Process; collapsed) ----
        # One "Voxel Resolution" slider drives the density and color grids;
        # the majorant grid is derived (min(6, voxel - 2)).
        if self._persisted_header("Grid Resolutions", "render_group_grid_resolutions"):
            _, voxel = imgui.slider_int(
                "Voxel Resolution (2^n)", ti.density_resolution_log2, 5, 10)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "EXPENSIVE: Determines the size of the finest details that "
                    "can be rendered by the path tracer")
            ti.density_resolution_log2 = voxel
            ti.color_resolution_log2 = voxel
            ti.majorant_resolution_log2 = min(6, voxel - 2)
            d = 2 ** ti.density_resolution_log2
            c = 2 ** ti.color_resolution_log2
            m = 2 ** ti.majorant_resolution_log2
            vram_mb = (d**3 * 4 + c**3 * 4 * 2 + m**3 * 4) / 1024**2
            imgui.text(f"VRAM: ~{vram_mb:.0f} MB")

        # ---- Post Process (Firefly clamp, shared; + shared Bloom) ----
        if self._persisted_header("Post Process", "render_group_opengl_postprocess"):
            self._render_firefly_clamp()
            self._render_bloom_controls()

    # ---------------------------------------------------- tracer helpers (moved)
    def _apply_tracer_preferences(self, ti):
        """Apply saved tracer preferences to a newly created TracerInterface."""
        p = self.state.preferences
        ti.colored_extinction = p.tracer.colored_extinction
        ti.sdf_enabled = p.tracer.sdf_enabled
        ti.extinction_rgb = list(p.tracer.extinction_rgb)
        ti.albedo_saturation = p.tracer.albedo_saturation
        ti.albedo_brightness = p.tracer.albedo_brightness
        ti.density_scale = p.tracer.density_scale
        ti.hg_g = p.tracer.hg_g
        ti.emission_strength = p.tracer.emission_strength
        # Sun + sky from the shared LightingPrefs slice (unified across renderers)
        ti.sun_direction = list(p.lighting.light_direction)
        ti.sun_color = list(p.lighting.light_color)
        ti.sun_intensity = p.lighting.light_intensity
        ti.sky_color_top = list(p.lighting.sky_color_top)
        ti.sky_color_bottom = list(p.lighting.sky_color_bottom)
        ti.sky_intensity = p.lighting.sky_intensity
        ti.sun_sampling = p.lighting.nee
        ti.photosphere = p.lighting.photosphere
        if ti.photosphere:
            ti._skybox_tex = ti._load_skybox()
            if ti._skybox_tex is None:
                ti.photosphere = False
        ti.max_bounces = p.tracer.max_bounces
        # rt-mode / capture-spp / resolution / firefly are shared RenderingPrefs
        # (pushed into ti each frame by the orchestrator); seed them here too so a
        # freshly-created interface starts consistent.
        ti.realtime_mode = p.rendering.rt_mode
        ti.num_samples = p.rendering.capture_spp
        ti.resolution_scale = p.rendering.render_resolution_scale
        ti.firefly_clamp = p.rendering.firefly_clamp
        ti.firefly_clamp_max = p.rendering.firefly_clamp_max
        ti.density_resolution_log2 = p.tracer.density_resolution_log2
        ti.color_resolution_log2 = p.tracer.color_resolution_log2
        ti.majorant_resolution_log2 = p.tracer.majorant_resolution_log2
