from .save_frame_gpu import save_frame_gpu, reset_gpu_frame_counter
from .ffmpeg_recorder import FFmpegVideoRecorder
from .paths import get_videos_dir
from datetime import datetime

class VidSaver:
    def __init__(self):
        self.active = False
        self.current_frame = 0
        self.recorder = None
        self.ssk_w = 2

    def frame(self, ctx, tex, max_frames=-1, ssk_w=2, filename_prefix=""):
        if not self.active:
            return

        # Calculate current output dimensions
        width, height = tex.size
        output_width = width // ssk_w
        output_height = height // ssk_w

        # Initialize recorder on first frame OR if dimensions/settings changed
        # Compare against input dimensions (not padded dimensions)
        if self.recorder is None or (
            self.recorder.input_width != output_width or
            self.recorder.input_height != output_height or
            self.ssk_w != ssk_w
        ):
            # If recorder exists but settings changed, close it and warn user
            if self.recorder is not None:
                print(f"WARNING: Recording settings changed mid-recording!")
                print(f"  Old: {self.recorder.input_width}x{self.recorder.input_height}, ssk={self.ssk_w}")
                print(f"  New: {output_width}x{output_height}, ssk={ssk_w}")
                print(f"  Finishing current video and starting new one...")
                self.recorder.close()

            # Create timestamped filename in Videos folder
            timestamp = datetime.now().strftime('%H-%M-%S')
            prefix = filename_prefix if filename_prefix else "animation"
            videos_dir = get_videos_dir()
            videos_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(videos_dir / f"{prefix}-{timestamp}.mp4")

            self.recorder = FFmpegVideoRecorder(
                width=output_width,
                height=output_height,
                fps=60,  # Default fps, can be made configurable
                output_path=output_path,
                realtime=False
            )
            self.ssk_w = ssk_w
            self.current_frame = 0  # Reset frame counter for new recording

        # Process frame with GPU (spatial supersampling only)
        # Temporal accumulation and gamma correction happen in FrameAssembler before this
        # Use return_array=True to get numpy array instead of saving PNG
        frame_array = save_frame_gpu(tex, ctx, supersample_k=ssk_w, return_array=True)

        # frame_array is always returned (no accumulation delay)
        self.recorder.write_frame_from_array(frame_array)
        self.current_frame += 1

        if max_frames > 0 and self.current_frame >= max_frames:
            self.finish()

    def finish(self):
        '''Save video and reset everything for another recording'''

        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

        reset_gpu_frame_counter()
        self.current_frame = 0
        self.active = False
                