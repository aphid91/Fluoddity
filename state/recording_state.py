from dataclasses import dataclass


@dataclass
class RecordingState:
    """Recording state machine data, owned by RecordingController (Step 10).

    Also nested (as a fresh, unused default) in UIState for the frame snapshot;
    the controller keeps its own live instance separate from that snapshot.
    """
    video_pending: bool = False           # Recording scheduled but not yet started
    video_scheduled_start_frame: int = 0  # Sim frame at which pending recording starts
    was_recording: bool = False           # Recording-active state on the previous frame
    user_speedmult: int = 1               # Saved speedmult to restore after recording
    user_motion_blur: bool = True         # Saved motion-blur toggle to restore
    user_blur_quality: int = 1            # Saved blur quality to restore
