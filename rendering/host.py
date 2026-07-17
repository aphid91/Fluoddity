"""RendererHost — owns the active OptiX renderer's lifecycle.

Step 7 extracts the ~300-line renderer create/sync/release/preview if/else
chain out of ``main.py orchestrate_frame()`` into this host. The host owns:

- **OptiX path tracer lifecycle**: lazy creation when enabled, VRAM release on
  toggle-off, per-frame error recovery (auto-disable on ``failed``).
- **Preference sync**: pushes ``preferences.optix`` + camera DOF onto the
  interface each frame (was ``main.py._sync_pathtracer_prefs``).
- **Preview lifecycle**: start / tick / cancel / result-invalidation.
- **Accumulation control**: reset on camera move (accumulate mode), force GAS
  rebuild after reset / config change.

Per-frame render *dispatch* still lives in the orchestrator for now (it reads
``host.optix`` to route the interface to the camera); a later phase moves the
image pipeline into the renderers. The volumetric tracer is created/driven by
the UI + orchestrator as before; the host only provides its teardown hook.
"""
from __future__ import annotations


class RendererHost:
    """Owns the active OptiX renderer and its lifecycle.

    The host is created once with the GL context. Each frame the orchestrator
    calls :meth:`update` with the current UI state and a few derived flags; the
    host reconciles the interface (create/release/sync) and exposes the active
    interface via :attr:`optix`.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        # The single OptiX renderer (lazy — created on first use when enabled).
        # rt_mode 0 (rasterize) is a preset on it; all modes share one interface.
        self.optix = None
        # Transition tracking (was on App)
        self._prev_optix_enabled = False
        self._prev_rt_mode = 0

    # ------------------------------------------------------------ OptiX lifecycle
    def _create_optix(self, ui_state) -> bool:
        """Lazily construct the path tracer. Returns True on success.

        On unavailability / failure, disables OptiX in ``ui_state`` and returns
        False (mirrors the old inline behavior).
        """
        try:
            from pathtracer_interface import PathTracerInterface
            if PathTracerInterface.is_available():
                self.optix = PathTracerInterface(self.ctx)
                print("OptiX path tracer initialized")
                return True
            ui_state.camera.optix_enabled = False
            print("OptiX not available — disabling")
            return False
        except Exception as e:
            ui_state.camera.optix_enabled = False
            print(f"OptiX init failed: {e}")
            return False

    def release_optix(self):
        """Release the OptiX interface's VRAM (idempotent)."""
        if self.optix is not None:
            self.optix.cleanup()
            self.optix = None

    def update(self, ui_state, *, is_recording: bool, needs_gas_rebuild: bool,
               camera_moved: bool):
        """Reconcile the OptiX interface for this frame.

        Returns ``pt_active`` — whether the path tracer is the active 3D
        renderer this frame (OptiX enabled and interface live).
        """
        rt_mode = ui_state.preferences.rendering.rt_mode
        pt_active = ui_state.camera.optix_enabled

        # Force full GAS rebuild after sim reset / config change
        if needs_gas_rebuild and self.optix is not None:
            self.optix.force_rebuild()

        # Release VRAM when toggled off (unless a preview result is pending)
        if (not pt_active and self._prev_optix_enabled and not is_recording
                and self.optix is not None):
            print("Releasing path tracer VRAM (OptiX disabled)")
            self.release_optix()

        # Lazy creation when toggled on
        if pt_active and self.optix is None:
            if not self._create_optix(ui_state):
                pt_active = False

        # Per-frame sync + error recovery
        if self.optix is not None:
            if self.optix.failed:
                print(f"OptiX auto-disabled: {self.optix.fail_reason}")
                self.optix = None
                ui_state.camera.optix_enabled = False
                pt_active = False
            elif pt_active:
                self.sync_optix_prefs(ui_state)
                ui_state.camera.pathtracer_gas_time_ms = self.optix.gas_time_ms
                ui_state.camera.pathtracer_render_time_ms = self.optix.render_time_ms
                ui_state.camera.pathtracer_sample_count = self.optix.sample_count

        # Accumulate mode: reset on camera movement
        if pt_active and rt_mode == 2 and self.optix is not None and camera_moved:
            self.optix.reset_accumulation()

        # Clear preview result on mode change, camera move, or sim reset
        if self.optix is not None:
            pv = self.optix
            if pv.preview_active or pv.preview_has_result:
                if rt_mode > 0:
                    # Preview is a rasterize feature; switched to a realtime PT mode
                    pv.cancel_preview()
                    pv._preview_has_result = False
                elif camera_moved or needs_gas_rebuild:
                    pv.cancel_preview()
                    pv._preview_has_result = False

        self._prev_optix_enabled = ui_state.camera.optix_enabled
        self._prev_rt_mode = rt_mode
        return pt_active

    def sync_optix_prefs(self, ui_state):
        """Push OptiX preferences + camera DOF onto the path tracer interface.

        Was ``main.py._sync_pathtracer_prefs``. Feeds shared geometry/lighting/
        material, path-trace-only controls, and the rasterize preset (rt_mode 0).
        """
        pt = self.optix
        if pt is None:
            return
        p = ui_state.preferences
        # Shared geometry + lighting
        pt.radius_scale = p.optix.sphere_radius_scale
        # Sun + sky from the shared LightingPrefs slice (unified across renderers)
        pt.sun_direction = tuple(p.lighting.light_direction)
        pt.sun_color = tuple(p.lighting.light_color)
        pt.sun_intensity = p.lighting.light_intensity
        pt.sky_color_top = tuple(p.lighting.sky_color_top)
        pt.sky_color_bottom = tuple(p.lighting.sky_color_bottom)
        pt.albedo_saturation = p.optix.albedo_saturation
        pt.albedo_brightness = p.optix.albedo_brightness
        pt.sphere_size_jitter = p.optix.sphere_size_jitter
        pt.use_curves = p.optix.use_curves
        pt.curve_length = p.optix.curve_length
        pt.curve_r0 = p.optix.curve_r0
        pt.curve_r1 = p.optix.curve_r1
        pt.sdf_enabled = p.optix.sdf_enabled
        pt.sun_sampling = p.lighting.nee
        pt.env_sky_nee = p.optix.pt_env_sky_nee
        pt.sun_exp = p.optix.pt_sun_exp
        pt.photosphere = p.lighting.photosphere
        # Path-trace-only
        pt.max_bounces = p.optix.pt_max_bounces
        pt.rr_start_depth = p.optix.pt_rr_start_depth
        pt.firefly_clamp = p.rendering.firefly_clamp
        pt.firefly_clamp_max = p.rendering.firefly_clamp_max
        pt.global_material = p.optix.pt_global_material
        pt.glossy_ior = p.optix.pt_glossy_ior
        pt.emission_intensity = p.optix.pt_emission_intensity
        pt.denoise_enabled = p.optix.pt_denoise_enabled
        pt.aperture = ui_state.camera.aperture
        pt.focal_plane_depth = ui_state.camera.focal_plane_depth
        # Rasterize preset (rt_mode 0). The preview always path-traces (that's
        # the point of the preview button — see start_preview call site), so
        # while a preview is active/showing its result, keep rasterize forced
        # off here too; otherwise this per-frame sync stomps the one-shot
        # override after the first sample.
        if pt.preview_active or pt.preview_has_result:
            pt.rasterize = False
        else:
            pt.rasterize = (p.rendering.rt_mode == 0)
        pt.ao_enabled = p.optix.ao_enabled
        pt.ao_num_rays = p.optix.ao_num_rays
        pt.ao_radius = p.optix.ao_radius
        pt.ambient = p.optix.ambient
        pt.ambient_color = tuple(p.optix.ambient_color)
        pt.rz_denoise_enabled = p.optix.rz_denoise_enabled
        pt.rz_depth_of_field = p.optix.rz_depth_of_field
        pt.rasterize_samples = p.optix.rz_samples
        # Pathtrace mode (0 = Off/rasterize, 1 = X spp, 2 = accumulate).
        pt.render_mode = 1 if p.rendering.rt_mode == 0 else p.rendering.rt_mode
        pt.realtime_samples = p.rendering.rt_samples

    def ensure_optix_for_preview(self, ui_state):
        """Lazy-create the path tracer specifically for a preview request.

        Returns the interface (or None if unavailable).
        """
        if self.optix is None:
            try:
                from pathtracer_interface import PathTracerInterface
                if PathTracerInterface.is_available():
                    self.optix = PathTracerInterface(self.ctx)
                    print("OptiX path tracer initialized (for preview)")
            except Exception as e:
                print(f"Path tracer init failed for preview: {e}")
        return self.optix
