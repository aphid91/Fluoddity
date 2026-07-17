"""RecordingController: owns the video-recording state machine (Step 10).

Extracted from the interleaved recording logic that lived in
``main.orchestrate_frame`` and ``command_handler``. Drives the
idle -> pending -> recording -> finished lifecycle: schedules a delayed
start, builds the active renderer's video strategy at record start, overrides
speedmult / motion-blur while recording (and restores them after), overrides AO
rays for rasterize-mode video, and fires a callback when recording ends by
reaching max_frames.
"""
from dataclasses import dataclass
from state.recording_state import RecordingState


@dataclass
class RecordingUpdate:
    """Result of RecordingController.update(), consumed by the orchestrator."""
    is_recording: bool
    tracer_video_active: bool
    optix_pt_video_active: bool
    active_video_strategy: object  # VideoStrategy or None


class RecordingController:
    """Owns the recording state machine and its RecordingState."""

    def __init__(self, video_service, camera, ui, sim):
        self.video_service = video_service
        self.camera = camera
        self.ui = ui
        self.sim = sim

        self.state = RecordingState()
        # The active renderer video strategy (built at recording start, cleared
        # at stop). Held here; the orchestrator reads it via update()'s result.
        self.active_video_strategy = None

    @property
    def video_pending(self):
        return self.state.video_pending

    @property
    def video_scheduled_start_frame(self):
        return self.state.video_scheduled_start_frame

    def toggle(self, ui_state):
        """Handle the recording toggle flag (with delayed-start support)."""
        st = self.state
        if st.video_pending:
            st.video_pending = False
            st.video_scheduled_start_frame = 0
        elif self.video_service.is_active():
            self.video_service.stop()
        else:
            rec = ui_state.preferences.recording
            video_end_frame = rec.video_end_frame
            video_simulation_frames = rec.max_frames * rec.motion_blur_samples
            scheduled_start_frame = video_end_frame - video_simulation_frames

            if video_end_frame == 0 or scheduled_start_frame <= self.sim.frame_count:
                self.video_service.start(stereo=ui_state.camera.stereogram)
            else:
                st.video_pending = True
                st.video_scheduled_start_frame = scheduled_start_frame

    def check_pending_start(self, ui_state):
        """Start a pending recording once the scheduled sim frame is reached."""
        st = self.state
        if st.video_pending and self.sim.frame_count >= st.video_scheduled_start_frame:
            st.video_pending = False
            st.video_scheduled_start_frame = 0
            self.video_service.start(stereo=ui_state.camera.stereogram)

    def update(self, ui_state, *, pathtracer_interface, sim_runner,
               on_finished_naturally):
        """Advance the recording lifecycle for this frame.

        Mutates ``ui_state.preferences.rendering`` to override / restore
        speedmult / motion-blur / blur-quality around recording, builds or drops
        the active video strategy, and fires ``on_finished_naturally`` when a
        recording ends by reaching max_frames.

        Returns a RecordingUpdate the orchestrator uses for the rest of the
        frame (video-frame branches, rt gating).
        """
        st = self.state
        p = ui_state.preferences

        # Renderer selection (ui_state.camera.optix_enabled == renderer is Optix):
        #   OpenGL + RT spp/accumulate  -> volumetric tracer (offline video)
        #   Optix  + rt_mode spp/accum  -> OptiX path tracer (offline video)
        #   OpenGL + RT Off / Optix rasterize -> normal per-frame assembly
        is_recording = self.video_service.is_active()
        optix_active = ui_state.camera.optix_enabled
        # Pathtrace mode is now a shared RenderingPrefs field (synced across
        # renderers), so it's authoritative for both.
        rt_mode = p.rendering.rt_mode
        tracer_video_active = (is_recording
                               and not optix_active
                               and rt_mode > 0)
        optix_pt_video_active = (is_recording
                                 and optix_active
                                 and rt_mode > 0
                                 and pathtracer_interface is not None)

        if is_recording and not st.was_recording:
            st.user_speedmult = p.rendering.speedmult
            st.user_motion_blur = p.rendering.motion_blur
            st.user_blur_quality = p.rendering.blur_quality
            # Build the renderer's video strategy when starting a recording
            if tracer_video_active:
                # Ensure TracerInterface is created
                if self.ui._tracer_interface is None:
                    from tracer_interface import TracerInterface
                    self.ui._tracer_interface = TracerInterface(self.camera.ctx)
                    self.ui._apply_tracer_preferences(self.ui._tracer_interface)
                # Sync DOF from camera state
                self.ui._tracer_interface.aperture = ui_state.camera.aperture
                self.ui._tracer_interface.focal_plane_depth = ui_state.camera.focal_plane_depth
                self.active_video_strategy = sim_runner.make_tracer_video_strategy(
                    self.ui._tracer_interface)
            elif optix_pt_video_active:
                # Build OptiX path tracer offline video strategy
                self.active_video_strategy = sim_runner.make_optix_pt_video_strategy(
                    pathtracer_interface)
            # Rasterize / RT-off mode needs no per-start override: Capture SPP
            # now caps the temporal motion-blur samples (applied in
            # SimulationRunner via the shared cadence-lock scheduler), and AO
            # rays keep coming from the user's own AO Rays slider (pushed each
            # frame by rendering.host).
        elif not is_recording and st.was_recording:
            p.rendering.speedmult = st.user_speedmult
            p.rendering.motion_blur = st.user_motion_blur
            p.rendering.blur_quality = st.user_blur_quality
            # Recording ended — drop the active video strategy
            self.active_video_strategy = None
            # Invalidate cached texture — the image pipeline recreates resources
            # when total_samples changes, releasing the old texture
            self.camera.assembled_texture = None
            # Pause simulation when recording ended by reaching max_frames
            if self.video_service.finished_naturally():
                on_finished_naturally(ui_state)

        if is_recording:
            if tracer_video_active:
                # Tracer mode: physics steps are managed by the video strategy
                p.rendering.speedmult = 1
                p.rendering.motion_blur = False
            elif optix_pt_video_active:
                # OptiX PT mode: physics steps managed by the video strategy
                p.rendering.speedmult = 1
                p.rendering.motion_blur = False
            else:
                p.rendering.speedmult = p.recording.motion_blur_samples
                p.rendering.motion_blur = p.recording.recording_motion_blur
                p.rendering.blur_quality = p.recording.recording_blur_quality

        st.was_recording = is_recording

        return RecordingUpdate(
            is_recording=is_recording,
            tracer_video_active=tracer_video_active,
            optix_pt_video_active=optix_pt_video_active,
            active_video_strategy=self.active_video_strategy,
        )
