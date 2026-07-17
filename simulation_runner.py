"""Simulation runner: physics stepping and frame assembly (normal render path).

Renderer-specific offline video paths now live behind the VideoStrategy
protocol (see rendering/video_strategies.py); this runner only builds them.
"""


class SimulationRunner:
    """Runs physics simulation steps with frame assembly and video recording.

    Handles both motion-blur (temporal accumulation) and non-motion-blur paths,
    deduplicating the shared physics stepping and frame assembly logic.
    """

    def __init__(self, sim, camera, video_service, command_handler, window,
                 controller_cam=None, plotting_manager=None):
        self.sim = sim
        self.camera = camera
        self.video_service = video_service
        self.command_handler = command_handler
        self.window = window
        self.controller_cam = controller_cam
        self.plotting_manager = plotting_manager

    def make_video_context(self):
        """Build the shared VideoContext used by renderer video strategies.

        Carries this runner's sim/camera/controller-cam/window plus the
        physics-step callback so a strategy can drive interleaved physics.
        """
        from rendering import VideoContext
        return VideoContext(
            sim=self.sim,
            camera=self.camera,
            controller_cam=self.controller_cam,
            window=self.window,
            run_physics_step=self._run_physics_step,
        )

    def make_tracer_video_strategy(self, tracer_interface):
        """Create a volumetric-tracer video strategy bound to this runner."""
        from rendering import TracerVideoStrategy
        return TracerVideoStrategy(self.make_video_context(), tracer_interface)

    def make_optix_pt_video_strategy(self, pt_interface):
        """Create an OptiX path-tracer offline video strategy bound to this runner."""
        from rendering import OptixPtVideoStrategy
        return OptixPtVideoStrategy(self.make_video_context(), pt_interface)

    def run_simulation_frame(self, ui_state,
                              screenshot_in_progress=False,
                              skip_view_generation=False):
        """Run simulation step(s) with frame assembly and video recording."""
        self._screenshot_in_progress = screenshot_in_progress
        speedmult = ui_state.preferences.rendering.speedmult
        motion_blur = ui_state.preferences.rendering.motion_blur

        # Handle clear canvas request (one-shot; reset after acting)
        if ui_state.request_clear_canvas:
            self.sim.clear_canvas()
            ui_state.request_clear_canvas = False

        # Build shared image-pipeline kwargs (used by both paths)
        assemble_kwargs = self._build_assemble_kwargs(ui_state)

        if motion_blur:
            self._run_with_motion_blur(
                ui_state, speedmult, assemble_kwargs,
                skip_view_generation=skip_view_generation
            )
        else:
            self._run_without_motion_blur(
                ui_state, speedmult, assemble_kwargs,
                skip_view_generation=skip_view_generation
            )

    def _build_assemble_kwargs(self, ui_state):
        """Build the kwargs dict for image_pipeline.assemble_frame().

        Shared between motion-blur and non-motion-blur paths; only
        total_samples / current_sample_index differ between them.
        """
        return dict(
            brightness=self.camera.BRIGHTNESS,
            tonemap_softness=ui_state.preferences.rendering.tonemap_softness,
            bloom_enabled=ui_state.preferences.bloom.enabled,
            bloom_threshold=ui_state.preferences.bloom.threshold,
            bloom_intensity=ui_state.preferences.bloom.intensity,
            bloom_radius=ui_state.preferences.bloom.radius,
        )

    def _run_physics_step(self, ui_state, step_index):
        """Run a single physics step and handle deferred entity selection.

        Args:
            step_index: Current step within the frame (0-based).
                        Entity selection only checked on step 0.
        """
        # Build generics tuple from preferences
        p = ui_state.preferences
        generics = (p.generics.generic0, p.generics.generic1, p.generics.generic2, p.generics.generic3,
                     p.generics.generic4, p.generics.generic5, p.generics.generic6, p.generics.generic7)

        self.sim.update(
            self.camera.ctx,
            generics=generics,
        )

        # Check for deferred entity selection only on first physics step
        if step_index == 0 and self.command_handler.has_pending_entity_selection:
            self.command_handler.try_complete_entity_selection(ui_state)

    def _process_assembled_frame(self, assembled_tex, ui_state):
        """Handle a completed finished frame: store it and feed to video recorder.

        The frame is markup-free (bloom applied inside the ImagePipeline;
        overlays composited separately for display), so it is exactly what the
        video recorder should capture.
        """
        if assembled_tex is None:
            return
        self.camera.assembled_texture = assembled_tex
        if self.video_service.is_active():
            # Always-3D orientation: no vertical flip (video comes out upright).
            flip_y = False
            self.video_service.process_frame(
                self.camera.ctx,
                assembled_tex,
                ui_state.preferences.recording.max_frames,
                ui_state.preferences.recording.supersample_k,
                ui_state.preferences.recording.filename_prefix,
                flip_y=flip_y
            )

    def _run_with_motion_blur(self, ui_state, speedmult,
                               assemble_kwargs,
                               skip_view_generation=False):
        """Motion blur path: temporal accumulation with multiple render calls.

        Uses the shared cadence-lock scheduler so the rasterize / RT-off path
        obeys the same capture contract as the traced backends: Capture SPP
        caps the total temporal samples, Blur Quality sets which physics frames
        are eligible (the stride), and the frame always advances the full
        physics span (``speedmult``) so playback speed stays locked. Each
        eligible slot may carry more than one sample when Capture SPP exceeds
        the eligible-slot count — the extra samples denoise the AO term (each
        render_frame re-samples AO with fresh rng).
        """
        from rendering.capture_cadence import build_capture_plan

        if self.plotting_manager is not None:
            self.plotting_manager.pre_physics_frame(self.sim.entity_update_program)

        rec = ui_state.preferences.recording
        # During recording, Capture SPP is authoritative for the sample count.
        # Outside recording (interactive motion blur), no Capture-SPP cap makes
        # sense, so fall back to one sample per eligible slot.
        recording = self.video_service.is_active()
        capture_spp = (ui_state.preferences.rendering.capture_spp if recording
                       else max(1, speedmult // max(rec.recording_blur_quality, 1)))

        plan = build_capture_plan(
            physics_rate=speedmult,
            capture_spp=capture_spp,
            blur_quality=ui_state.preferences.rendering.blur_quality,
            motion_blur=True,
        )

        physics_done = 0
        render_sample_index = 0
        for slot in plan.slots:
            # Advance physics to this cadence slot.
            while physics_done < slot.physics_before:
                self._run_physics_step(ui_state, physics_done)
                if self.plotting_manager is not None:
                    self.plotting_manager.notify_physics_step()
                physics_done += 1

            if not skip_view_generation:
                for _ in range(slot.samples):
                    raw_view_tex = self.camera.generate_view_texture()
                    assembled_tex = self.camera.image_pipeline.assemble_frame(
                        raw_view_tex,
                        total_samples=plan.total_samples,
                        current_sample_index=render_sample_index,
                        **assemble_kwargs
                    )
                    render_sample_index += 1
                    self._process_assembled_frame(assembled_tex, ui_state)

        # Run out any remaining physics so the frame advances the full span
        # (playback-speed lock), matching the traced backends.
        while physics_done < plan.total_physics:
            self._run_physics_step(ui_state, physics_done)
            if self.plotting_manager is not None:
                self.plotting_manager.notify_physics_step()
            physics_done += 1

        if self.plotting_manager is not None:
            self.plotting_manager.post_assembly_frame()

    def _run_without_motion_blur(self, ui_state, speedmult,
                                  assemble_kwargs,
                                  skip_view_generation=False):
        """Non-motion-blur path: multiple physics steps, single render call."""
        if self.plotting_manager is not None:
            self.plotting_manager.pre_physics_frame(self.sim.entity_update_program)

        for step in range(speedmult):
            self._run_physics_step(ui_state, step)
            if self.plotting_manager is not None:
                self.plotting_manager.notify_physics_step()

        if not skip_view_generation:
            raw_view_tex = self.camera.generate_view_texture()

            assembled_tex = self.camera.image_pipeline.assemble_frame(
                raw_view_tex,
                total_samples=1,
                current_sample_index=0,
                **assemble_kwargs
            )

            self._process_assembled_frame(assembled_tex, ui_state)

        if self.plotting_manager is not None:
            self.plotting_manager.post_assembly_frame()

