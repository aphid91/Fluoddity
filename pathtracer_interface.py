"""Path tracer interface: bridges Fluoddity's entity buffer and camera to the OptiX path tracer.

Manages PathTracerRenderer lifecycle (lazy creation, cleanup), entity buffer
change detection, and per-frame error recovery. Supports realtime rendering
in multiple modes (X spp with reset, accumulate without reset) and offline
(multi-substep motion-blur) rendering, plus a progressive preview system.
"""
from __future__ import annotations

import numpy as np
import moderngl

from optix_pathtracer import PathTracerRenderer
from optix_pathtracer.renderer import _camera_basis_from_vectors
from optix_pathtracer import sdf_scene


class PathTracerInterface:
    """Manages the PathTracerRenderer for the 3D camera view.

    Created lazily on first render request. All GPU resources are allocated
    through the shared ModernGL context. If an OptiX error occurs during
    rendering, the interface auto-disables and reports failure via the
    ``failed`` property.
    """

    # Class-level cache: survives interface recreation across mode cycles
    _cached_photosphere_image: dict | None = None

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._renderer: PathTracerRenderer | None = None
        self._display_tex: moderngl.Texture | None = None

        # Entity buffer tracking (for change detection)
        self._entity_buffer_glo: int = 0
        self._entity_count: int = 0

        # Error recovery state
        self._failed: bool = False
        self._fail_reason: str = ""

        # Timing (updated each frame from renderer)
        self._gas_time_ms: float = 0.0
        self._render_time_ms: float = 0.0

        # --- Public attributes (wired to UI via preferences) ---

        # Lighting (shared across rasterize + path-trace modes via unified preferences)
        self.sun_direction: tuple[float, float, float] = (0.577, 0.577, 0.577)
        self.sun_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.sun_intensity: float = 1.0
        self.sun_sampling: bool = True
        self.sky_color_top: tuple[float, float, float] = (0.45, 0.62, 0.85)
        self.sky_color_bottom: tuple[float, float, float] = (0.08, 0.08, 0.10)

        # Geometry (shared)
        self.radius_scale: float = 1.0
        self.sphere_size_jitter: float = 0.0

        # Camera
        self.aperture: float = 0.0
        self.focal_plane_depth: float = 10.0

        # Bounce control
        self.max_bounces: int = 8
        self.rr_start_depth: int = 3
        self.firefly_clamp: bool = True
        self.firefly_clamp_max: float = 50.0

        # Material
        self.global_material: int = 0  # 0=Lambert, 1=Glossy, 2=Mirror
        self.glossy_ior: float = 1.5

        # Denoiser (separate toggle per lighting model)
        self.denoise_enabled: bool = False       # path-trace modes
        self.rz_denoise_enabled: bool = False     # rasterize mode

        # Rasterize preset (merged sphere renderer): single-hit direct lighting
        # + AO-modulated fake ambient. Active when rasterize is True (rt_mode 0).
        self.rasterize: bool = False
        self.ao_enabled: bool = False
        self.ao_num_rays: int = 2
        self.ao_radius: float = 0.5
        self.ambient: float = 0.12
        self.ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.rz_depth_of_field: bool = False  # allow thin-lens DOF in rasterize mode
        self.rasterize_samples: int = 1  # samples/frame accumulated in rasterize mode
        self._ao_frame_index: int = 0

        # Emissive particles
        self.emission_intensity: float = 10.0

        # Curve primitives
        self.use_curves: bool = False
        self.curve_length: float = 1.0
        self.curve_r0: float = 1.0
        self.curve_r1: float = 0.5

        # Albedo color controls (shared)
        self.albedo_saturation: float = 0.8
        self.albedo_brightness: float = 1.0

        # SDF scene
        self.sdf_enabled: bool = False

        # Environment sky NEE
        self.env_sky_nee: bool = False
        # Cos-lobe sky sun sharpness (UI "Sun Sharpness")
        self.sun_exp: float = 15.0

        # Photosphere
        self.photosphere: bool = False
        self._photosphere_image_data: dict | None = None  # cached loaded image
        self._photosphere_avg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Physics step tracking: set by orchestrator each frame before render
        self.physics_steps: int = 0

        # RT mode controls
        self.render_mode: int = 1  # 1=X spp (reset each frame), 2=accumulate
        self.realtime_samples: int = 1  # samples/frame for mode 1

        # Preview state
        self._preview_active: bool = False
        self._preview_target_spp: int = 0
        self._preview_last_spp: int = 0
        self._preview_has_result: bool = False
        # Camera state for preview (captured at start)
        self._preview_cam: dict | None = None

    # --------------------------------------------------------- photosphere loading

    def _load_photosphere_image(self) -> dict | None:
        """Load skybox.jpg, convert sRGB->linear, downscale 2x, return as f16 RGBA.

        Result is cached at class level so it survives interface recreation.
        """
        # Return class-level cache if available
        if PathTracerInterface._cached_photosphere_image is not None:
            return PathTracerInterface._cached_photosphere_image

        import os
        try:
            from PIL import Image
        except ImportError:
            print("Photosphere: PIL not available")
            return None

        skybox_path = os.path.join(os.path.dirname(__file__),
                                   "volrender", "textures", "skybox.jpg")
        if not os.path.exists(skybox_path):
            print(f"Photosphere: {skybox_path} not found")
            return None

        try:
            img = Image.open(skybox_path).convert('RGB')

            # Downscale 2x
            new_w = img.width // 2
            new_h = img.height // 2
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Convert to float32 RGB
            data = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(
                new_h, new_w, 3).astype(np.float32) / 255.0

            # sRGB to linear
            linear = np.where(data < 0.04045,
                              data / 12.92,
                              ((data + 0.055) / 1.055) ** 2.4)

            # Compute average color (on linear data)
            avg_color = tuple(linear.mean(axis=(0, 1)).tolist())

            # Pad to RGBA and convert to float16
            rgba = np.ones((new_h, new_w, 4), dtype=np.float32)
            rgba[:, :, :3] = linear
            rgba_f16 = rgba.astype(np.float16)

            # Ensure contiguous C-order array
            rgba_f16 = np.ascontiguousarray(rgba_f16)

            result = {
                'data': rgba_f16,
                'width': new_w,
                'height': new_h,
                'avg_color': avg_color,
            }
            PathTracerInterface._cached_photosphere_image = result
            return result
        except Exception as e:
            print(f"Photosphere: failed to load skybox: {e}")
            return None

    def _ensure_photosphere_texture(self):
        """Lazy-load photosphere image and upload CUDA texture if needed."""
        if not self.photosphere:
            return

        if self._photosphere_image_data is None:
            self._photosphere_image_data = self._load_photosphere_image()
            if self._photosphere_image_data is not None:
                self._photosphere_avg_color = self._photosphere_image_data['avg_color']
            else:
                self.photosphere = False
                return

        if self._renderer is not None and not self._renderer._photo_loaded:
            self._renderer.set_photosphere_texture(self._photosphere_image_data)

    # ------------------------------------------------------------ rasterize helpers

    def _rasterize_kwargs(self) -> dict:
        """Render-kwargs fragment carrying the rasterize preset + AO settings.

        The effective ambient tint folds the scalar `ambient` into the color
        picker so both UI controls are live. `ao_frame_index` decorrelates AO
        jitter across frames.
        """
        amb = (
            self.ambient_color[0] * self.ambient,
            self.ambient_color[1] * self.ambient,
            self.ambient_color[2] * self.ambient,
        )
        return dict(
            rasterize=self.rasterize,
            ao_enabled=self.ao_enabled,
            ao_num_rays=self.ao_num_rays,
            ao_radius=self.ao_radius,
            ao_frame_index=self._ao_frame_index,
            ambient_color=amb,
        )

    def _effective_aperture(self) -> float:
        """Aperture to use this frame.

        Path-trace modes always honor the camera aperture. Rasterize mode is
        pinhole by default (aperture 0), unless its Depth of Field toggle is on
        — then it uses the full thin-lens raygen like path-trace mode.
        """
        if self.rasterize and not self.rz_depth_of_field:
            return 0.0
        return self.aperture

    def _effective_denoise(self) -> bool:
        """Denoise toggle for the active lighting model."""
        return self.rz_denoise_enabled if self.rasterize else self.denoise_enabled

    # ------------------------------------------------------------------ core API

    def render_frame(
        self,
        entity_buffer: moderngl.Buffer,
        entity_count: int,
        cam_pos: np.ndarray,
        cam_dir: np.ndarray,
        cam_up: np.ndarray,
        fov: float,
        width: int,
        height: int,
        accum_slot: int = 0,
    ) -> moderngl.Texture | None:
        """Render one path-traced frame based on current render_mode.

        Mode 1 (X spp): Reset accumulation, trace realtime_samples, denoise
            if enabled, tonemap. Returns fresh image each frame.
        Mode 2 (Accumulate): No reset, trace 1 sample, denoise if enabled,
            tonemap running average. Call reset_accumulation() on camera move.

        GAS rebuild scheduling uses self.physics_steps (set by orchestrator)
        to track when entities have moved.

        Args:
            entity_buffer: ModernGL buffer (SSBO binding 0, 8-float stride).
            entity_count: Number of entities in the buffer.
            cam_pos: Camera position (3,) from ControllerCam.pos.
            cam_dir: Unit look direction (3,) from ControllerCam.dir.
            cam_up: Unit up vector (3,) from ControllerCam.up.
            fov: Vertical field of view in degrees.
            width: Output image width.
            height: Output image height.

        Returns:
            moderngl.Texture (rgba8) or None if OptiX unavailable/failed.
        """
        if self._failed:
            return None

        try:
            return self._render_frame_inner(
                entity_buffer, entity_count,
                cam_pos, cam_dir, cam_up, fov,
                width, height, accum_slot,
            )
        except Exception as e:
            self._failed = True
            self._fail_reason = str(e)
            print(f"PathTracer render error (auto-disabling): {e}")
            self.cleanup()
            return None

    def _render_frame_inner(
        self,
        entity_buffer, entity_count,
        cam_pos, cam_dir, cam_up, fov,
        width, height, accum_slot=0,
    ):
        """Inner render logic, called from render_frame() with error wrapping."""
        # 1. Lazy initialization
        if self._renderer is None:
            self._renderer = PathTracerRenderer(
                self.ctx, entity_buffer, entity_count
            )
            self._entity_buffer_glo = int(entity_buffer.glo)
            self._entity_count = entity_count

        # 2. Entity buffer change detection
        current_glo = int(entity_buffer.glo)
        if (current_glo != self._entity_buffer_glo
                or entity_count != self._entity_count):
            self._renderer.update_entity_buffer(entity_buffer, entity_count)
            self._entity_buffer_glo = current_glo
            self._entity_count = entity_count

        # 2b. Ensure photosphere texture is loaded if enabled
        self._ensure_photosphere_texture()

        # 3. Camera basis conversion
        aspect = width / max(height, 1)
        eye, U, V, W = _camera_basis_from_vectors(
            cam_pos, cam_dir, cam_up, fov, aspect
        )

        # Compute normalized cam_right and cam_up for DOF lens sampling
        cam_dir_arr = np.asarray(cam_dir, dtype=np.float32)
        cam_up_arr = np.asarray(cam_up, dtype=np.float32)
        cam_right_vec = np.cross(cam_dir_arr, cam_up_arr)
        cam_right_vec = cam_right_vec / max(np.linalg.norm(cam_right_vec), 1e-8)
        cam_up_vec = np.cross(cam_right_vec, cam_dir_arr)
        cam_up_vec = cam_up_vec / max(np.linalg.norm(cam_up_vec), 1e-8)

        # 4. Normalize sun direction
        sd = np.array(self.sun_direction, dtype=np.float64)
        length = max(np.linalg.norm(sd), 1e-8)
        sun_dir_norm = tuple((sd / length).astype(np.float32))

        # 5. Build render kwargs
        render_kwargs = dict(
            sun_direction=sun_dir_norm,
            sun_intensity=self.sun_intensity,
            sun_color=self.sun_color,
            sun_sampling=self.sun_sampling,
            exposure=1.0,  # fixed; brightness controlled via frame_assembly
            sky_color_top=self.sky_color_top,
            sky_color_bottom=self.sky_color_bottom,
            aperture=self._effective_aperture(),
            focal_plane_depth=self.focal_plane_depth,
            cam_right=cam_right_vec,
            cam_up=cam_up_vec,
            max_bounces=self.max_bounces,
            rr_start_depth=self.rr_start_depth,
            firefly_clamp=self.firefly_clamp,
            firefly_clamp_max=self.firefly_clamp_max,
            global_material=self.global_material,
            glossy_ior=self.glossy_ior,
            albedo_saturation=self.albedo_saturation,
            albedo_brightness=self.albedo_brightness,
            sphere_size_jitter=self.sphere_size_jitter,
            sdf_enabled=self.sdf_enabled,
            sdf_aabb_min=sdf_scene.SDF_AABB_MIN,
            sdf_aabb_max=sdf_scene.SDF_AABB_MAX,
            emission_intensity=self.emission_intensity,
            use_curves=self.use_curves,
            curve_length=self.curve_length,
            curve_r0=self.curve_r0,
            curve_r1=self.curve_r1,
            env_sky_nee=self.env_sky_nee,
            sun_exp=self.sun_exp,
            photosphere=self.photosphere,
            photosphere_avg_r=self._photosphere_avg_color[0] if self.photosphere else 0.0,
            photosphere_avg_g=self._photosphere_avg_color[1] if self.photosphere else 0.0,
            photosphere_avg_b=self._photosphere_avg_color[2] if self.photosphere else 0.0,
            **self._rasterize_kwargs(),
        )

        # 6. Dispatch based on render mode.
        # Rasterize and X-spp both reset each frame and accumulate N samples;
        # accumulate mode (2) keeps accumulating a single sample per frame.
        if self.rasterize:
            reset = True
            num_samples = self.rasterize_samples
        elif self.render_mode == 1:
            reset = True
            num_samples = self.realtime_samples
        else:
            reset = False
            num_samples = 1

        self._display_tex = self._renderer.render_realtime(
            width, height, eye, U, V, W,
            radius_scale=self.radius_scale,
            denoise_enabled=self._effective_denoise(),
            reset=reset,
            num_samples=num_samples,
            physics_steps=self.physics_steps,
            accum_slot=accum_slot,
            **render_kwargs,
        )
        self._ao_frame_index += 1

        # 7. Read timing from renderer
        self._gas_time_ms = self._renderer.last_gas_ms
        self._render_time_ms = self._renderer.last_render_ms

        return self._display_tex

    # ------------------------------------------------------------- preview API

    def start_preview(self, target_spp: int,
                      entity_buffer: moderngl.Buffer,
                      entity_count: int,
                      cam_pos: np.ndarray,
                      cam_dir: np.ndarray,
                      cam_up: np.ndarray,
                      fov: float,
                      width: int,
                      height: int) -> None:
        """Start a progressive preview render at the given target SPP.

        Each subsequent call to tick_preview() traces 1 sample and
        accumulates. The preview builds up over multiple app frames
        to keep the program responsive.
        """
        if self._failed:
            return

        # Lazy initialization
        if self._renderer is None:
            self._renderer = PathTracerRenderer(
                self.ctx, entity_buffer, entity_count
            )
            self._entity_buffer_glo = int(entity_buffer.glo)
            self._entity_count = entity_count

        # Entity buffer change detection
        current_glo = int(entity_buffer.glo)
        if (current_glo != self._entity_buffer_glo
                or entity_count != self._entity_count):
            self._renderer.update_entity_buffer(entity_buffer, entity_count)
            self._entity_buffer_glo = current_glo
            self._entity_count = entity_count

        # Build GAS
        self._renderer.build_accel(
            self.radius_scale, self.sphere_size_jitter,
            self.sdf_enabled, sdf_scene.SDF_AABB_MIN if self.sdf_enabled else None,
            sdf_scene.SDF_AABB_MAX if self.sdf_enabled else None,
            self.use_curves, self.curve_length, self.curve_r0, self.curve_r1)
        self._renderer.reset_accumulation()

        # Camera basis
        aspect = width / max(height, 1)
        eye, U, V, W = _camera_basis_from_vectors(
            cam_pos, cam_dir, cam_up, fov, aspect
        )
        cam_dir_arr = np.asarray(cam_dir, dtype=np.float32)
        cam_up_arr = np.asarray(cam_up, dtype=np.float32)
        cam_right_vec = np.cross(cam_dir_arr, cam_up_arr)
        cam_right_vec = cam_right_vec / max(np.linalg.norm(cam_right_vec), 1e-8)
        cam_up_vec = np.cross(cam_right_vec, cam_dir_arr)
        cam_up_vec = cam_up_vec / max(np.linalg.norm(cam_up_vec), 1e-8)

        sd = np.array(self.sun_direction, dtype=np.float64)
        slen = max(np.linalg.norm(sd), 1e-8)
        sun_dir_norm = tuple((sd / slen).astype(np.float32))

        self._preview_cam = dict(
            eye=eye, U=U, V=V, W=W,
            width=width, height=height,
            sun_dir_norm=sun_dir_norm,
            cam_right=cam_right_vec,
            cam_up=cam_up_vec,
        )
        self._preview_target_spp = target_spp
        self._preview_active = True
        self._preview_has_result = False

    def tick_preview(self) -> bool:
        """Trace 1 sample for the preview. Returns True when done.

        When done and denoise is enabled, runs the denoiser on the
        fully accumulated result.
        """
        if not self._preview_active or self._renderer is None:
            return True
        if self._preview_cam is None:
            return True

        c = self._preview_cam
        w, h = c['width'], c['height']

        render_kwargs = dict(
            sun_direction=c['sun_dir_norm'],
            sun_intensity=self.sun_intensity,
            sun_color=self.sun_color,
            sun_sampling=self.sun_sampling,
            exposure=1.0,
            sky_color_top=self.sky_color_top,
            sky_color_bottom=self.sky_color_bottom,
            aperture=self._effective_aperture(),
            focal_plane_depth=self.focal_plane_depth,
            cam_right=c['cam_right'],
            cam_up=c['cam_up'],
            max_bounces=self.max_bounces,
            rr_start_depth=self.rr_start_depth,
            firefly_clamp=self.firefly_clamp,
            firefly_clamp_max=self.firefly_clamp_max,
            global_material=self.global_material,
            glossy_ior=self.glossy_ior,
            albedo_saturation=self.albedo_saturation,
            albedo_brightness=self.albedo_brightness,
            sphere_size_jitter=self.sphere_size_jitter,
            sdf_enabled=self.sdf_enabled,
            sdf_aabb_min=sdf_scene.SDF_AABB_MIN,
            sdf_aabb_max=sdf_scene.SDF_AABB_MAX,
            emission_intensity=self.emission_intensity,
            use_curves=self.use_curves,
            curve_length=self.curve_length,
            curve_r0=self.curve_r0,
            curve_r1=self.curve_r1,
            env_sky_nee=self.env_sky_nee,
            sun_exp=self.sun_exp,
            photosphere=self.photosphere,
            photosphere_avg_r=self._photosphere_avg_color[0] if self.photosphere else 0.0,
            photosphere_avg_g=self._photosphere_avg_color[1] if self.photosphere else 0.0,
            photosphere_avg_b=self._photosphere_avg_color[2] if self.photosphere else 0.0,
            **self._rasterize_kwargs(),
        )

        done = self._renderer._sample_count >= self._preview_target_spp

        if not done:
            # Trace 1 sample without reset, no denoise during accumulation
            self._display_tex = self._renderer.render_realtime(
                w, h, c['eye'], c['U'], c['V'], c['W'],
                radius_scale=self.radius_scale,
                denoise_enabled=False,
                reset=False,
                num_samples=1,
                flip_y=False,  # Preview displayed directly, no intermediate blit
                **render_kwargs,
            )
            self._ao_frame_index += 1
            done = self._renderer._sample_count >= self._preview_target_spp

        if done:
            # Final frame: denoise if enabled, then tonemap
            if self._effective_denoise():
                self._display_tex = self._renderer.render_realtime(
                    w, h, c['eye'], c['U'], c['V'], c['W'],
                    radius_scale=self.radius_scale,
                    denoise_enabled=True,
                    reset=False,
                    num_samples=0,  # no additional samples, just denoise+tonemap
                    flip_y=False,  # Preview displayed directly, no intermediate blit
                    **render_kwargs,
                )
            self._preview_last_spp = self._renderer._sample_count
            self._preview_active = False
            self._preview_has_result = True
            return True

        return False

    def cancel_preview(self) -> None:
        """Cancel the active preview render."""
        self._preview_active = False

    @property
    def preview_active(self) -> bool:
        """True if a preview render is in progress."""
        return self._preview_active

    @property
    def preview_samples_done(self) -> int:
        """Number of samples accumulated so far in the preview."""
        if self._renderer is not None:
            return self._renderer._sample_count
        return 0

    @property
    def preview_target_spp(self) -> int:
        """Target SPP for the current/last preview."""
        return self._preview_target_spp

    @property
    def preview_has_result(self) -> bool:
        """True if a completed preview result is available."""
        return self._preview_has_result

    @property
    def preview_last_spp(self) -> int:
        """SPP of the last completed preview."""
        return self._preview_last_spp

    # ------------------------------------------------------------- entity pick

    def pick(self, ray_origin, ray_dir):
        """Trace one ray into the last-built GAS; return (entity_id, depth) or None.

        Returns None on miss, SDF hit, no renderer/GAS yet, or any OptiX error
        (so the caller can fall back to the CPU nearest-particle picker). Uses
        the most recently rendered frame's GAS, which is current while OptiX is
        the live renderer.
        """
        if self._renderer is None or self._failed:
            return None
        try:
            return self._renderer.pick(ray_origin, ray_dir)
        except Exception as e:
            print(f"OptiX pick failed: {e}")
            return None

    # --------------------------------------------------------- accumulation API

    def reset_accumulation(self) -> None:
        """Reset the accumulation buffer (call on camera move in accumulate mode).

        Clears all accumulation slots so both stereogram eyes reset together
        (non-stereo just clears slot 0).
        """
        if self._renderer is not None:
            self._renderer.reset_all_accum_slots()

    @property
    def sample_count(self) -> int:
        """Current number of accumulated samples."""
        if self._renderer is not None:
            return self._renderer._sample_count
        return 0

    # -------------------------------------------------------------- offline API

    def start_offline_render(
        self,
        entity_buffer: moderngl.Buffer,
        entity_count: int,
        width: int,
        height: int,
        total_substeps: int,
        spp_per_substep: int,
        accum_slot: int = 0,
    ) -> None:
        """Begin an offline motion-blur render.

        Initializes the renderer (if needed), handles entity buffer changes,
        builds the GAS, and starts the offline accumulation sequence.

        After calling this, call offline_substep() for each temporal sub-step
        (updating entity positions between calls), then finish_offline_render()
        to get the final tonemapped image.
        """
        if self._failed:
            raise RuntimeError(f"PathTracer failed: {self._fail_reason}")

        # Lazy initialization
        if self._renderer is None:
            self._renderer = PathTracerRenderer(
                self.ctx, entity_buffer, entity_count
            )
            self._entity_buffer_glo = int(entity_buffer.glo)
            self._entity_count = entity_count

        # Entity buffer change detection
        current_glo = int(entity_buffer.glo)
        if (current_glo != self._entity_buffer_glo
                or entity_count != self._entity_count):
            self._renderer.update_entity_buffer(entity_buffer, entity_count)
            self._entity_buffer_glo = current_glo
            self._entity_count = entity_count

        # Full GAS build before starting offline render
        self._renderer.build_accel(
            self.radius_scale, self.sphere_size_jitter,
            self.sdf_enabled, sdf_scene.SDF_AABB_MIN if self.sdf_enabled else None,
            sdf_scene.SDF_AABB_MAX if self.sdf_enabled else None,
            self.use_curves, self.curve_length, self.curve_r0, self.curve_r1)

        # Begin offline accumulation
        self._renderer.render_offline_begin(
            width, height, total_substeps, spp_per_substep,
            denoise_enabled=self._effective_denoise(),
            accum_slot=accum_slot,
        )

    def offline_substep(
        self,
        cam_pos: np.ndarray,
        cam_dir: np.ndarray,
        cam_up: np.ndarray,
        fov: float,
        accum_slot: int = 0,
        spp: int | None = None,
    ) -> None:
        """Trace one temporal sub-step of an offline motion-blur render.

        The caller must update entity positions (via physics step) and ensure
        the entity buffer reflects the new state BEFORE calling. The GAS is
        refitted internally to match the updated positions.

        For stereogram video, pass the eye camera (cam_pos/cam_dir offset per
        eye) and that eye's ``accum_slot``.
        """
        if self._renderer is None:
            raise RuntimeError(
                "Call start_offline_render() before offline_substep()."
            )

        # Camera basis conversion
        w = self._renderer._offline_width
        h = self._renderer._offline_height
        aspect = w / max(h, 1)
        eye, U, V, W = _camera_basis_from_vectors(
            cam_pos, cam_dir, cam_up, fov, aspect
        )

        # DOF basis
        cam_dir_arr = np.asarray(cam_dir, dtype=np.float32)
        cam_up_arr = np.asarray(cam_up, dtype=np.float32)
        cam_right_vec = np.cross(cam_dir_arr, cam_up_arr)
        cam_right_vec = cam_right_vec / max(np.linalg.norm(cam_right_vec), 1e-8)
        cam_up_vec = np.cross(cam_right_vec, cam_dir_arr)
        cam_up_vec = cam_up_vec / max(np.linalg.norm(cam_up_vec), 1e-8)

        # Normalize sun direction
        sd = np.array(self.sun_direction, dtype=np.float64)
        length = max(np.linalg.norm(sd), 1e-8)
        sun_dir_norm = tuple((sd / length).astype(np.float32))

        self._renderer.render_offline_substep(
            eye, U, V, W,
            radius_scale=self.radius_scale,
            accum_slot=accum_slot,
            spp=spp,
            sun_direction=sun_dir_norm,
            sun_intensity=self.sun_intensity,
            sun_color=self.sun_color,
            sun_sampling=self.sun_sampling,
            exposure=1.0,
            sky_color_top=self.sky_color_top,
            sky_color_bottom=self.sky_color_bottom,
            aperture=self._effective_aperture(),
            focal_plane_depth=self.focal_plane_depth,
            cam_right=cam_right_vec,
            cam_up=cam_up_vec,
            max_bounces=self.max_bounces,
            rr_start_depth=self.rr_start_depth,
            firefly_clamp=self.firefly_clamp,
            firefly_clamp_max=self.firefly_clamp_max,
            global_material=self.global_material,
            glossy_ior=self.glossy_ior,
            albedo_saturation=self.albedo_saturation,
            albedo_brightness=self.albedo_brightness,
            sphere_size_jitter=self.sphere_size_jitter,
            sdf_enabled=self.sdf_enabled,
            sdf_aabb_min=sdf_scene.SDF_AABB_MIN,
            sdf_aabb_max=sdf_scene.SDF_AABB_MAX,
            emission_intensity=self.emission_intensity,
            use_curves=self.use_curves,
            curve_length=self.curve_length,
            curve_r0=self.curve_r0,
            curve_r1=self.curve_r1,
            env_sky_nee=self.env_sky_nee,
            sun_exp=self.sun_exp,
            photosphere=self.photosphere,
            photosphere_avg_r=self._photosphere_avg_color[0] if self.photosphere else 0.0,
            photosphere_avg_g=self._photosphere_avg_color[1] if self.photosphere else 0.0,
            photosphere_avg_b=self._photosphere_avg_color[2] if self.photosphere else 0.0,
            **self._rasterize_kwargs(),
        )
        self._ao_frame_index += 1

        # Read timing
        self._gas_time_ms = self._renderer.last_gas_ms
        self._render_time_ms = self._renderer.last_render_ms

    def finish_offline_render(self, flip_y=True, accum_slot=0,
                              end_session=True) -> moderngl.Texture | None:
        """Denoise (if enabled) and tonemap the fully-accumulated offline frame.

        Must be called after all sub-steps are complete.

        Args:
            flip_y: If True, flip Y axis for OpenGL convention (default).
            accum_slot: Accumulation slot to resolve (stereogram: per eye).
            end_session: If False, keep the offline session open so a second
                eye can still resolve (stereogram). Set True on the last eye.

        Returns:
            moderngl.Texture (rgba16f/rgba8) with the final image, or None.
            NOTE: the returned texture is the renderer's shared scratch texture;
            for stereogram, copy it out before resolving the next eye.
        """
        if self._renderer is None:
            return None

        try:
            self._display_tex = self._renderer.render_offline_finish(
                flip_y=flip_y, accum_slot=accum_slot, end_session=end_session,
            )
            return self._display_tex
        except Exception as e:
            self._failed = True
            self._fail_reason = str(e)
            print(f"PathTracer offline finish error: {e}")
            return None

    # ----------------------------------------------------------- shader reload

    def reload_shaders(self):
        """Hot-reload CUDA path tracer source (recompile PTX + rebuild pipeline)."""
        if self._renderer is not None:
            try:
                self._renderer.reload_shaders()
            except Exception as e:
                print(f"PathTracer shader reload failed: {e}")

    # --------------------------------------------------------------- scheduling

    def force_rebuild(self):
        """Force a full GAS rebuild on the next frame.

        Call after sim reset or any event that moves all entities at once.
        """
        if self._renderer is not None:
            self._renderer._gas_handle = None

    # ---------------------------------------------------------------- properties

    @property
    def display_texture(self) -> moderngl.Texture | None:
        """The rgba8 rendered texture for display, or None if no render yet."""
        return self._display_tex

    @property
    def gas_time_ms(self) -> float:
        """Time in ms for the most recent GAS build/refit."""
        return self._gas_time_ms

    @property
    def render_time_ms(self) -> float:
        """Time in ms for the most recent render (trace + denoise + tonemap)."""
        return self._render_time_ms

    @property
    def failed(self) -> bool:
        """True if the path tracer encountered a fatal error and auto-disabled."""
        return self._failed

    @property
    def fail_reason(self) -> str:
        """Human-readable reason for the failure, or empty string."""
        return self._fail_reason

    # ---------------------------------------------------------------- lifecycle

    def cleanup(self):
        """Release all OptiX/CUDA resources."""
        if self._renderer is not None:
            try:
                self._renderer.cleanup()
            except Exception:
                pass
            self._renderer = None
        self._display_tex = None
        self._entity_buffer_glo = 0
        self._entity_count = 0
        self._preview_active = False
        self._preview_cam = None

    # -------------------------------------------------------------- availability

    @staticmethod
    def is_available() -> bool:
        """Check if OptiX/CUDA/RTX hardware is available."""
        return PathTracerRenderer.is_available()
