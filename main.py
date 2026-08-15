import glfw
import moderngl
import time
import numpy as np
from camera import Camera
from sim import Sim, SIZE_OF_ENTITY_STRUCT
from ui import UI
from services import RuleManager, EntityPicker, VideoRecorderService, ConfigSaver, ArrowDebugService, MultiLoadService, StreamlineService, AudioService
from services.field_handler import FieldHandler
from services.parameter_lock_service import ParameterLockService
from utilities.paths import initialize_user_data, get_user_physics_configs_dir, get_app_physics_configs_dir, get_screenshots_dir
from state import (load_preferences, save_preferences, SimState,
                   load_streamline_state, load_audio_state)
from state.audio_state import AUDIO_BLOCK
from services.audio_capture import AudioCapture, mux_audio_into_video
from pathlib import Path

# Video is written at a fixed 60fps (see utilities/vid_saver.py).
VIDEO_FPS = 60
from command_handler import CommandHandler
from simulation_runner import SimulationRunner
from camera_input import process_camera_input
from controller_input import ControllerCam, process_controller_input, find_joystick
from utilities.advanced_drawing import AdvancedDrawingProcessor


class App:
    """Main application orchestrator.

    Coordinates all components each frame: reads UI state, delegates commands
    to CommandHandler, physics to SimulationRunner, and manages recording/
    screenshot state machines.
    """

    def __init__(self):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        self.window = glfw.create_window(800, 600, "Fluoddity", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync

        # Initialize ModernGL
        self.ctx = moderngl.create_context()
        self.ctx.gc_mode = 'auto'

        # Always on top ONLY FOR WHEN LIVE EDITING THE SHADERS, NOT IN DISTRIBUTION
        #glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)

        # Initialize user data directory (creates Documents/Fluoddity on first run)
        initialize_user_data()

        # Load preferences first to get world_size
        loaded_prefs = load_preferences()

        # Create components (no cross-references between UI and sim/camera)
        self.sim = Sim(self.ctx, world_size=loaded_prefs.world_size, canvas_aspect_ratio=loaded_prefs.canvas_aspect_ratio)
        self.camera = Camera(self.ctx, self.sim, self.window)
        self.ui = UI(self.window, self.ctx, self.sim.view_option_labels)

        # Apply loaded preferences to UI
        self.ui.state.preferences = loaded_prefs
        self.ui._last_applied_world_size = loaded_prefs.world_size
        self.ui.state.streamline = load_streamline_state()
        self.ui.state.audio = load_audio_state()

        # Create services (Orchestrator owns these)
        self.rule_manager = RuleManager()
        entity_stride = SIZE_OF_ENTITY_STRUCT // 4
        self.entity_picker = EntityPicker(self.sim.get_entity_buffer(), entity_stride)
        self.video_service = VideoRecorderService()
        self.config_saver = ConfigSaver()
        self.arrow_debug_service = ArrowDebugService(self.ctx)
        self.streamline_service = StreamlineService(self.ctx)
        self.audio_service = AudioService(self.ctx)
        self._streamline_last_time = time.time()  # Tracer's own clock
        # User's own dispatch schedule, held while audio locks the clock.
        self._streamline_sched_backup = None
        self.audio_capture = None  # Offline audio capture while recording
        self._last_render_dt = 1.0 / 60.0  # Smoothed render frame time
        self.multi_load_service = MultiLoadService()
        self.advanced_drawing_processor = AdvancedDrawingProcessor(self.ctx)
        self.ui.multi_load_service = self.multi_load_service
        self.ui.advanced_drawing_processor = self.advanced_drawing_processor
        self.ui.streamline_service = self.streamline_service

        # Physics configs directories
        self.app_configs_dir = get_app_physics_configs_dir()
        self.user_configs_dir = get_user_physics_configs_dir()
        self.user_configs_dir.mkdir(exist_ok=True)

        # Create delegated handlers
        self.param_lock_service = ParameterLockService()
        self.field_handler = FieldHandler(
            self.advanced_drawing_processor, self.sim,
            param_lock_service=self.param_lock_service)
        self.ui.param_lock_service = self.param_lock_service
        self.command_handler = CommandHandler(
            self.sim, self.camera, self.ui, self.rule_manager,
            self.entity_picker, self.video_service, self.config_saver,
            self.multi_load_service, self.user_configs_dir,
            field_handler=self.field_handler,
            param_lock_service=self.param_lock_service,
            streamline_service=self.streamline_service,
            audio_service=self.audio_service
        )
        # Xbox controller (FPS camera for shader-driven field)
        self.controller_cam = ControllerCam()
        self.joystick_state = {'joystick_id': find_joystick(), 'prev_buttons': []}

        self.sim_runner = SimulationRunner(
            self.sim, self.camera, self.video_service,
            self.command_handler, self.window,
            advanced_drawing_processor=self.advanced_drawing_processor,
            controller_cam=self.controller_cam
        )
        self.sim_runner.pre_record_hook = self._composite_streamlines_into_frame

        # Frame timing
        self.last_update_time = time.time()

        # Track user's desired settings (for restoration after recording)
        self.user_speedmult = 1
        self.was_recording = False
        self.user_motion_blur = True
        self.user_blur_quality = 1

        # Screenshot state machine
        self.screenshot_pending = False
        self.screenshot_in_progress = False
        self.screenshot_saved_settings = {}

        # Track previous view option for camera repositioning when leaving tiling mode
        self.prev_view_option = 0

        # Ensure _Default.json exists and load it
        self._ensure_default_config()
        self._load_default_config()
        self.sim.reload()
        self.sim.reset()

    def _ensure_default_config(self):
        """Ensure _Default.json exists in physics_configs directory. Create it if missing."""
        default_path = self.app_configs_dir / "Core/_Default.json"
        if not default_path.exists():
            default_state = SimState()
            zero_rule = np.zeros((10, 8), dtype=np.float32)
            config = self.config_saver.create_config(default_state, zero_rule)
            self.config_saver.save_to_file(config, default_path)
            print(f"Created default config: {default_path}")

    def _load_default_config(self):
        """Load _Default.json on startup."""
        default_path = self.app_configs_dir / "Core/_Default.json"
        config = self.config_saver.load_from_file(default_path)
        if config is not None:
            rule = self.config_saver.apply_config(config, self.ui.state.sim)
            self.rule_manager.push_rule(rule, self.ui.state.sim.rule_seed)
            self.sim.apply_rule(rule)
            print(f"Loaded default config from {default_path}")
            self.ui.update_physics_defaults("_Default")
        else:
            print(f"Failed to load default config from {default_path}")

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.orchestrate_frame()
            glfw.swap_buffers(self.window)

        self.cleanup()

    def orchestrate_frame(self):
        """Main orchestration logic - reads UI state, coordinates components."""

        # 1. Get current UI state
        ui_state = self.ui.get_state()
        tiling_mode = (ui_state.sim.current_view_option == 2)

        # 2. Process one-shot commands
        result = self.command_handler.process_commands(ui_state, tiling_mode)
        if result == 'screenshot_pending' and not self.screenshot_pending and not self.screenshot_in_progress:
            self.screenshot_pending = True

        # 3. Process continuous input (camera movement)
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        process_camera_input(ui_state, self.window, self.ui.keybindings,
                             self.sim.view_tex, dt)
        process_controller_input(self.controller_cam, self.joystick_state, dt)

        # 3.2. Check if pending video should start
        cmd = self.command_handler
        if cmd.video_pending and self.sim.frame_count >= cmd.video_scheduled_start_frame:
            cmd.video_pending = False
            cmd.video_scheduled_start_frame = 0
            self.video_service.start()

        # 3.5. Screenshot state machine
        if self.screenshot_pending and not self.screenshot_in_progress:
            self.screenshot_pending = False
            self.screenshot_in_progress = True
            self.screenshot_saved_settings = {
                'speedmult': ui_state.preferences.speedmult,
                'blur_quality': ui_state.preferences.blur_quality,
                'motion_blur': ui_state.preferences.motion_blur,
                'going': ui_state.sim.going,
            }
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.blur_quality = 1
            ui_state.preferences.motion_blur = True
            if not ui_state.sim.going:
                ui_state.sim.going = True

        # 4. Lock physics frequency to video recorder frequency if recording
        is_recording = self.video_service.is_active()

        if is_recording and not self.was_recording:
            self.user_speedmult = ui_state.preferences.speedmult
            self.user_motion_blur = ui_state.preferences.motion_blur
            self.user_blur_quality = ui_state.preferences.blur_quality
            self._begin_audio_capture(ui_state)
        elif not is_recording and self.was_recording:
            ui_state.preferences.speedmult = self.user_speedmult
            ui_state.preferences.motion_blur = self.user_motion_blur
            ui_state.preferences.blur_quality = self.user_blur_quality
            self._finish_audio_capture()

        if is_recording:
            ui_state.preferences.speedmult = ui_state.preferences.motion_blur_samples
            ui_state.preferences.motion_blur = ui_state.preferences.recording_motion_blur
            ui_state.preferences.blur_quality = ui_state.preferences.recording_blur_quality

        self.was_recording = is_recording

        # 5. Apply state to components
        if ui_state.request_camera_reset:
            ui_state.camera.position[:] = [0.0, 0.0]
            ui_state.camera.zoom = 1.0
        self.sim.apply_state(ui_state.sim)
        self.sim.apply_camera_state(ui_state.camera)
        self.camera.apply_state(ui_state.camera)
        self.multi_load_service.apply_state(ui_state.multi_load)
        self.camera.BRIGHTNESS = ui_state.preferences.brightness

        # 5.0.1 Force/Strafe field view modes: override view_tex with field texture
        if ui_state.sim.current_view_option in (3, 4):
            field_tex = self.advanced_drawing_processor.field_texture
            if field_tex is not None:
                self.sim.view_tex = field_tex
            else:
                # Field texture not initialized yet — fall back to canvas view
                ui_state.sim.current_view_option = 0

        # 5.1. Multi-load conflict prevention
        if ui_state.multi_load.multi_load_enabled:
            ui_state.sim.parameter_sweeps_enabled = False
            ui_state.preferences.mouse_mode = "Draw Trail"
            if self.param_lock_service.enabled:
                self.param_lock_service.reset()
                ui_state.preferences.parameter_locks_enabled = False

        # 5.2. Sync parameter lock master toggle
        self.param_lock_service.enabled = ui_state.preferences.parameter_locks_enabled

        # 5.5. Calculate sweep reticle info
        sweep_reticle_x, sweep_reticle_y, sweep_reticle_visible = self.sim.get_sweep_reticle_position()

        if is_recording or self.screenshot_in_progress:
            sweep_reticle_visible = False

        width, height = glfw.get_framebuffer_size(self.window)
        screen_aspect = width / height if height > 0 else 1.0

        if sweep_reticle_visible:
            screen_x, screen_y = self.camera.tex_to_screen(
                (sweep_reticle_x, sweep_reticle_y),
                self.sim.view_tex.size
            )
            sweep_reticle_x = screen_x / width
            sweep_reticle_y = screen_y / height

        sweep_mode = ui_state.sim.parameter_sweeps_enabled
        sweep_reticle_pos = (sweep_reticle_x, sweep_reticle_y)

        # Reposition camera when leaving tiling mode
        if self.prev_view_option == 2 and ui_state.sim.current_view_option != 2:
            ui_state.camera.position[0] = np.fmod(ui_state.camera.position[0] + 100.0, 2.0) - 1.0
            ui_state.camera.position[1] = np.fmod(ui_state.camera.position[1] + 100.0, 2.0) - 1.0
        self.prev_view_option = ui_state.sim.current_view_option

        # 6. Run simulation if going
        if ui_state.sim.going:
            self.sim_runner.run_simulation_frame(
                ui_state, sweep_mode, sweep_reticle_pos, sweep_reticle_visible,
                screen_aspect, ui_state.sim.watercolor_mode,
                tiling_mode=tiling_mode,
                screenshot_in_progress=self.screenshot_in_progress
            )

        # 6.5. Screenshot save and settings restoration
        if self.screenshot_in_progress:
            self._save_screenshot(ui_state)

        # 7. Render camera view
        self._render_camera_view(ui_state, sweep_mode, sweep_reticle_pos,
                                  sweep_reticle_visible, screen_aspect, tiling_mode)

        # 7.5. Render arrow debug overlay if enabled
        if ui_state.preferences.debug_arrows:
            width, height = glfw.get_framebuffer_size(self.window)
            adv_prefs = ui_state.preferences
            adv_active = adv_prefs.advanced_drawing_enabled
            field_tex = self.advanced_drawing_processor.field_texture
            # Use field_texture for force/strafe targets, canvas for trails
            if adv_active and not adv_prefs.advanced_draw_canvas and field_tex is not None:
                arrow_texture = field_tex
                arrow_resolution = field_tex.size
                use_zw = adv_prefs.advanced_draw_strafe_field
            else:
                arrow_texture = self.sim.can
                arrow_resolution = self.sim.can.size
                use_zw = False
            self.arrow_debug_service.render(
                canvas_texture=arrow_texture,
                cam_pos=tuple(self.camera.position),
                cam_zoom=self.camera.zoom,
                canvas_resolution=arrow_resolution,
                window_size=(width, height),
                arrow_sensitivity=ui_state.preferences.arrow_sensitivity,
                use_zw_channels=use_zw,
            )

        # 7.6. Step the streamline tracer and draw its overlay.
        # Audio keeps the tracer running even with the overlay hidden, since
        # the voice is generated by the same dispatches.
        if (ui_state.streamline.enabled or ui_state.audio.enabled
                or self.audio_capture is not None):
            width, height = glfw.get_framebuffer_size(self.window)
            streamline = ui_state.streamline
            # Seed at the cursor: screen pixels -> texture [0,1] -> world [-1,1]
            tex_x, tex_y = self.camera.screen_to_tex(
                ui_state.mouse_pos, tex_size=self.sim.can.size
            )
            cursor_seed = (tex_x * 2.0 - 1.0, tex_y * 2.0 - 1.0)

            # Middle-click pins the seed in place (and unpins it again), so the
            # streamlines stay put while the field evolves under them.
            if ui_state.middle_click_this_frame:
                streamline.seed_pinned = not streamline.seed_pinned
                if streamline.seed_pinned:
                    streamline.pinned_seed = cursor_seed

            seed = streamline.pinned_seed if streamline.seed_pinned else cursor_seed

            # The tracer runs on its own clock, so it keeps its rate whether or
            # not the physics is paused and regardless of the frame rate.
            now = time.time()
            dt = now - self._streamline_last_time
            self._streamline_last_time = now
            # Clamp so a hitch or a breakpoint cannot inject a huge dt.
            dt = min(max(dt, 0.0), 0.25)
            if dt > 0.0:
                # Smoothed, so one hitch does not swing the interpolation rate.
                self._last_render_dt = self._last_render_dt * 0.9 + dt * 0.1

            # Audio drives the tracer clock when enabled, so open/close the
            # stream before stepping.
            audio = ui_state.audio
            # An offline capture owns the tracer for the duration, so live
            # playback stays down even though audio.enabled is still set - it
            # records the user's intent to resume once the render finishes.
            want_live = audio.enabled and self.audio_capture is None
            if want_live and not self.audio_service.active:
                if not self.audio_service.start(audio):
                    audio.enabled = False  # Start failed; report via last_error
            elif not want_live and self.audio_service.active:
                self.audio_service.stop()

            if want_live and self.audio_service.active:
                # Locked to the audio contract: one block of samples per
                # dispatch, at the device's rate. Stash the user's own values
                # first and restore them on release, so the locked numbers are
                # never what gets persisted or left behind.
                if self._streamline_sched_backup is None:
                    self._streamline_sched_backup = (
                        streamline.steps_per_dispatch, streamline.dispatch_hz)
                streamline.steps_per_dispatch = AUDIO_BLOCK
                streamline.dispatch_hz = audio.sample_rate / AUDIO_BLOCK
            elif self._streamline_sched_backup is not None:
                (streamline.steps_per_dispatch,
                 streamline.dispatch_hz) = self._streamline_sched_backup
                self._streamline_sched_backup = None

            if self.audio_capture is not None:
                # Offline: the video frame decides how much audio to render,
                # and those same dispatches advance the tracer. Running the
                # wall-clock path as well would double-step the particles.
                self._render_audio_for_frame(ui_state, seed)
            else:
                self.streamline_service.update(
                    canvas_texture=self.sim.can,
                    seed_world=seed,
                    settings=streamline,
                    dt=dt,
                    audio_service=self.audio_service,
                    audio_settings=audio,
                    prev_texture=self._prev_canvas(),
                    samples_per_physics_step=self._samples_per_physics_step(ui_state),
                )
                if self.audio_service.active:
                    self.audio_service.pump(audio)
            self.audio_service.update_telemetry(audio)
            if streamline.enabled:
                self.streamline_service.draw(
                    cam_pos=tuple(self.camera.position),
                    cam_zoom=self.camera.zoom,
                    canvas_resolution=self.sim.can.size,
                    window_size=(width, height),
                    settings=streamline,
                )
        else:
            # Keep the clock from banking time while the overlay is off.
            self._streamline_last_time = time.time()

        # 8. Update UI display info and render
        self.ui.update_display_info({
            'time': self.sim.time,
            'frame_count': self.sim.frame_count,
            'tex_size': self.sim.view_tex.size,
            'recording_active': self.video_service.is_active(),
            'video_pending': cmd.video_pending,
            'video_scheduled_start_frame': cmd.video_scheduled_start_frame,
        })
        self.ui.render()

    def _render_camera_view(self, ui_state, sweep_mode, sweep_reticle_pos,
                             sweep_reticle_visible, screen_aspect, tiling_mode):
        """Render the camera view to screen."""
        draw_trail_mode = ui_state.preferences.mouse_mode == "Draw Trail"

        width, height = glfw.get_framebuffer_size(self.window)
        mouse_x_norm = ui_state.mouse_pos[0] / width if width > 0 else 0.5
        mouse_y_norm = ui_state.mouse_pos[1] / height if height > 0 else 0.5
        mouse_screen_coords = (mouse_x_norm, mouse_y_norm)

        self.camera.render(
            sim_going=ui_state.sim.going,
            current_view_option=ui_state.sim.current_view_option,
            sweep_mode=sweep_mode,
            sweep_reticle_pos=sweep_reticle_pos,
            sweep_reticle_visible=sweep_reticle_visible,
            screen_aspect=screen_aspect,
            watercolor_mode=ui_state.sim.watercolor_mode,
            ink_weight=ui_state.sim.ink_weight,
            draw_trail_mode=draw_trail_mode,
            draw_size=ui_state.preferences.draw_size,
            mouse_screen_coords=mouse_screen_coords,
            exposure=ui_state.preferences.exposure,
            tiling_mode=tiling_mode,
            tonemap_softness=ui_state.preferences.tonemap_softness,
            bloom_enabled=ui_state.preferences.bloom_enabled,
            bloom_threshold=ui_state.preferences.bloom_threshold,
            bloom_intensity=ui_state.preferences.bloom_intensity,
            bloom_radius=ui_state.preferences.bloom_radius
        )

    def _save_screenshot(self, ui_state):
        """Save screenshot and restore settings."""
        if self.camera.assembled_texture is not None:
            from utilities.save_frame_gpu import save_frame_gpu
            import datetime
            import os
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = ui_state.preferences.filename_prefix or "screenshot"
            filename = save_frame_gpu(
                self.camera.assembled_texture,
                self.ctx,
                supersample_k=ui_state.preferences.supersample_k,
                return_array=False
            )
            if filename and os.path.exists(filename):
                screenshots_dir = get_screenshots_dir()
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                new_filename = screenshots_dir / f"{prefix}_{timestamp}.png"
                os.rename(filename, new_filename)
                print(f"Screenshot saved: {new_filename}")

        # Restore saved settings
        ui_state.preferences.speedmult = self.screenshot_saved_settings['speedmult']
        ui_state.preferences.blur_quality = self.screenshot_saved_settings['blur_quality']
        ui_state.preferences.motion_blur = self.screenshot_saved_settings['motion_blur']
        ui_state.sim.going = self.screenshot_saved_settings['going']
        self.screenshot_in_progress = False
        self.screenshot_saved_settings = {}

    def _prev_canvas(self):
        """The canvas as of the previous physics step.

        Double buffering is forced on, so the texture that is not currently
        being read holds the previous frame. Returns None if that is not the
        case, and the tracer then falls back to no interpolation.
        """
        try:
            other = self.sim.can_textures[1 - self.sim.can_read_index]
        except (AttributeError, IndexError):
            return None
        return other if other is not self.sim.can else None

    def _samples_per_physics_step(self, ui_state) -> float:
        """Integration steps the tracer runs per canvas update.

        The canvas only changes once per physics step, so this is how many
        audio samples span one field update - the interval the field is
        interpolated across.
        """
        speedmult = max(1, int(ui_state.preferences.speedmult))
        if self.audio_capture is not None:
            # Offline: a video frame is worth sample_rate/fps samples, spread
            # over speedmult physics steps.
            return (ui_state.audio.sample_rate / VIDEO_FPS) / speedmult
        if ui_state.audio.enabled and self.audio_service.active:
            # Realtime: physics advances speedmult steps per render frame.
            render_fps = max(1.0, 1.0 / max(self._last_render_dt, 1e-3))
            return ui_state.audio.sample_rate / (speedmult * render_fps)
        # Visual only: the tracer's own schedule sets the pace.
        streamline = ui_state.streamline
        steps_per_sec = max(1.0, streamline.dispatch_hz * streamline.steps_per_dispatch)
        render_fps = max(1.0, 1.0 / max(self._last_render_dt, 1e-3))
        return steps_per_sec / (speedmult * render_fps)

    def _composite_streamlines_into_frame(self, assembled_tex, ui_state):
        """Draw the streamline overlay into a frame bound for the encoder."""
        streamline = ui_state.streamline
        if not (streamline.enabled and streamline.render_to_video):
            return
        fbo = self.ctx.framebuffer(color_attachments=[assembled_tex])
        prev = self.ctx.fbo
        try:
            fbo.use()
            # Draw in the texture's own space: the assembled frame is the
            # whole canvas, so no camera transform applies here.
            self.streamline_service.draw(
                cam_pos=(0.0, 0.0),
                cam_zoom=1.0,
                canvas_resolution=assembled_tex.size,
                window_size=assembled_tex.size,
                settings=streamline,
            )
        finally:
            if prev is not None:
                prev.use()
            fbo.release()

    def _begin_audio_capture(self, ui_state):
        """Start offline audio capture if the recording asked for it."""
        self.audio_capture = None
        audio = ui_state.audio
        if not audio.record_audio:
            return
        # Realtime playback and an offline render cannot both drive the
        # tracer, so stand the live stream down for the duration.
        if self.audio_service.active:
            self.audio_service.stop()
        self.audio_service.reset_offline(audio)
        self.audio_capture = AudioCapture(audio.sample_rate, VIDEO_FPS)
        print(f"Recording audio at {audio.sample_rate} Hz "
              f"({self.audio_capture.samples_per_frame:.1f} samples/frame)")

    def _render_audio_for_frame(self, ui_state, seed_world):
        """Render the audio one recorded video frame is worth.

        A video frame is 1/VIDEO_FPS of output however many physics steps
        went into it, so the sample count per frame is fixed; speedmult only
        changes how far the field moves between samples.
        """
        cap = self.audio_capture
        if cap is None:
            return
        blocks = cap.blocks_owed(1)
        if blocks <= 0:
            return
        for block in self.streamline_service.render_audio_blocks(
                self.sim.can, seed_world, ui_state.streamline,
                self.audio_service, ui_state.audio, blocks,
                prev_texture=self._prev_canvas(),
                samples_per_physics_step=self._samples_per_physics_step(ui_state)):
            cap.add_block(block)

    def _finish_audio_capture(self):
        """Write the captured audio and mux it into the finished video."""
        cap = self.audio_capture
        self.audio_capture = None
        if cap is None or not cap.has_audio():
            return
        video_path = self.video_service.last_output_path
        if not video_path:
            print("Audio captured but no video path to mux into.")
            return
        wav_path = Path(str(video_path).replace('.mp4', '.wav'))
        try:
            cap.write_wav(wav_path)
        except Exception as e:
            print(f"Failed to write audio track: {e}")
            return
        print(f"Muxing {cap.duration:.2f}s of audio into {video_path}")
        mux_audio_into_video(video_path, wav_path)

    def cleanup(self):
        # Save preferences before cleanup
        ui_state = self.ui.get_state()
        # If we exit while audio holds the clock, persist the user's own
        # schedule rather than the locked audio values.
        if self._streamline_sched_backup is not None:
            (ui_state.streamline.steps_per_dispatch,
             ui_state.streamline.dispatch_hz) = self._streamline_sched_backup
        save_preferences(ui_state.preferences,
                         streamline=ui_state.streamline,
                         audio=ui_state.audio)

        self.advanced_drawing_processor.cleanup()
        self.audio_service.cleanup()
        self.streamline_service.cleanup()
        self.video_service.cleanup()
        self.ui.cleanup()
        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()
