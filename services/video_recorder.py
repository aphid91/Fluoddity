from utilities.vid_saver import VidSaver


class VideoRecorderService:
    """Wraps VidSaver, provides clean interface for Orchestrator."""

    def __init__(self):
        self.recorder = VidSaver()

    def is_active(self) -> bool:
        """Check if recording is active."""
        return self.recorder.active

    @property
    def last_output_path(self):
        """Path of the most recently finished video, for audio muxing."""
        return self.recorder.last_output_path

    def start(self) -> None:
        """Start recording."""
        if not self.recorder.active:
            self.recorder.active = True

    def stop(self) -> None:
        """Stop recording and save video."""
        if self.recorder.active:
            self.recorder.finish()

    def toggle(self) -> None:
        """Toggle recording state."""
        if self.recorder.active:
            self.stop()
        else:
            self.start()

    def process_frame(self, ctx, texture, max_frames: int, ssk_w: int, filename_prefix: str = "") -> None:
        """Process a frame if recording is active.

        Args:
            texture: Already assembled and gamma-corrected texture
            max_frames: Maximum frames to record
            ssk_w: Spatial supersample kernel width
            filename_prefix: Custom filename prefix (empty = use "animation")
        """
        self.recorder.frame(ctx, texture, max_frames, ssk_w, filename_prefix)

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.recorder.active:
            self.recorder.finish()
