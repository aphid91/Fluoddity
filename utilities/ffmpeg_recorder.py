import subprocess
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path
from utilities.paths import get_videos_dir


def find_ffmpeg():
    """
    Find ffmpeg executable, checking bundled location first, then system PATH.

    Returns:
        str: Path to ffmpeg executable

    Raises:
        FileNotFoundError: If ffmpeg cannot be found
    """
    # Check if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check for bundled ffmpeg
        bundle_dir = Path(sys._MEIPASS)
        ffmpeg_path = bundle_dir / 'ffmpeg.exe'
        if ffmpeg_path.exists():
            return str(ffmpeg_path)

    # Check system PATH
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # Not found - provide helpful error message
    raise FileNotFoundError(
        "FFmpeg not found!\n\n"
        "Video recording requires FFmpeg. Please either:\n"
        "1. Install FFmpeg and add it to your system PATH, or\n"
        "2. Place ffmpeg.exe in the same folder as this application\n\n"
        "Download FFmpeg from: https://ffmpeg.org/download.html"
    )

class FFmpegVideoRecorder:
    """
    Video recorder that pipes frames directly to ffmpeg without intermediate PNG files.
    More efficient than the old PNG-based approach.

    Video Quality Settings:
    - CRF (Constant Rate Factor): Controls quality/file-size tradeoff
      * Range: 0-51 (lower = better quality, larger file)
      * 18 = visually lossless for most content (high quality mode)
      * 23 = default/good balance (realtime mode)
      * 28 = acceptable quality for previews
      * Each +6 roughly doubles file size

    - Preset: Controls encoding speed vs compression efficiency
      * ultrafast, superfast, veryfast, faster, fast (realtime default)
      * medium (default preset)
      * slow (high quality default), slower, veryslow
      * Slower presets = better compression (smaller files) at same quality

    To increase quality: Lower CRF (e.g., 15) and/or use slower preset (e.g., 'veryslow')
    To decrease quality/size: Raise CRF (e.g., 28) and/or use faster preset (e.g., 'veryfast')
    """

    def __init__(self, width, height, fps=60, output_path=None, realtime=True, debug_log=False):
        """
        Initialize ffmpeg video recorder.

        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            output_path: Path for output video. If None, generates timestamped filename
            realtime: If True, use faster encoding. If False, use higher quality
            debug_log: If True, save ffmpeg output to a log file for debugging
        """
        # Store original dimensions
        self.input_width = width
        self.input_height = height

        # H.264 requires dimensions divisible by 2 (ideally 16)
        # Pad to even numbers
        self.width = width + (width % 2)
        self.height = height + (height % 2)

        self.needs_padding = (self.width != width or self.height != height)

        if self.needs_padding:
            print(f"Note: Padding video from {width}x{height} to {self.width}x{self.height} (H.264 requires even dimensions)")

        self.fps = fps

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%H-%M-%S')
            videos_dir = get_videos_dir()
            videos_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(videos_dir / f"animation-{timestamp}.mp4")

        self.output_path = output_path

        # Quality settings
        if realtime:
            preset = 'fast'
            crf = 23  # Reasonable quality
        else:
            preset = 'slow'
            crf = 18  # Higher quality (lower = better)

        # Find ffmpeg executable (bundled or system PATH)
        ffmpeg_cmd = find_ffmpeg()

        # Start ffmpeg process
        # Optionally capture ffmpeg output to log file for debugging
        if debug_log:
            import os
            self.stderr_log_path = output_path.replace('.mp4', '_ffmpeg.log')
            self.stderr_log = open(self.stderr_log_path, 'w')
            stderr_dest = self.stderr_log
        else:
            self.stderr_log = None
            self.stderr_log_path = None
            stderr_dest = subprocess.DEVNULL

        self.ffmpeg = subprocess.Popen([
            ffmpeg_cmd, '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.width}x{self.height}',  # Use padded dimensions
            '-framerate', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',  # Compatible with most players
            output_path
        ], stdin=subprocess.PIPE, stderr=stderr_dest, stdout=subprocess.DEVNULL)

        self.frame_count = 0

    def write_frame_from_texture(self, accumulation_texture):
        """
        Read frame from ModernGL texture and write to video.

        Args:
            accumulation_texture: moderngl.Texture with the rendered frame
        """
        # Read from texture
        data = accumulation_texture.read()

        # Convert to numpy array
        frame = np.frombuffer(data, dtype=np.float32).reshape(self.height, self.width, 4)

        # Drop alpha channel and convert to uint8
        frame = frame[:, :, :3]
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        # Flip vertically (OpenGL -> image coordinates)
        frame = np.flipud(frame)

        # Write to ffmpeg
        self.ffmpeg.stdin.write(frame.tobytes())
        self.frame_count += 1

    def write_frame_from_array(self, frame_array):
        """
        Write frame from numpy array directly to video.

        Args:
            frame_array: numpy array (height, width, 3) of uint8 RGB data,
                        already flipped and in correct orientation
        """
        # Validate frame dimensions match input expectations
        if frame_array.shape != (self.input_height, self.input_width, 3):
            raise ValueError(
                f"Frame dimensions {frame_array.shape} don't match recorder "
                f"input dimensions ({self.input_height}, {self.input_width}, 3). "
                f"Did the window size change during recording?"
            )

        # Pad frame if needed for H.264 encoding
        if self.needs_padding:
            padded = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            padded[:self.input_height, :self.input_width, :] = frame_array
            frame_array = padded

        # Check if ffmpeg process is still alive
        if self.ffmpeg.poll() is not None:
            # Try to read the error log if debug logging is enabled
            error_msg = f"FFmpeg process has terminated unexpectedly with return code {self.ffmpeg.returncode}"
            if self.stderr_log:
                self.stderr_log.flush()
                self.stderr_log.close()
            if self.stderr_log_path:
                try:
                    with open(self.stderr_log_path, 'r') as f:
                        log_contents = f.read()
                        if log_contents:
                            error_msg += f"\n\nFFmpeg error log:\n{log_contents}"
                except:
                    pass
            else:
                error_msg += "\n\n(Enable debug_log=True to see detailed ffmpeg output)"
            raise RuntimeError(error_msg)

        try:
            self.ffmpeg.stdin.write(frame_array.tobytes())
            self.frame_count += 1
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"Failed to write frame to ffmpeg: {e}. "
                f"The ffmpeg process may have crashed or the pipe is broken."
            ) from e

    def close(self):
        """Close the video file and finish encoding."""
        if self.ffmpeg.stdin:
            try:
                # Flush any buffered data before closing
                self.ffmpeg.stdin.flush()
                self.ffmpeg.stdin.close()
            except Exception as e:
                print(f"Error closing ffmpeg stdin: {e}")

        # Wait for ffmpeg to finish with a reasonable timeout
        # Give it more time if we have a lot of frames
        timeout = max(15.0, self.frame_count / 100.0)  # At least 15s, or 1s per 100 frames
        try:
            returncode = self.ffmpeg.wait(timeout=timeout)

            # Close stderr log if it exists
            if self.stderr_log:
                self.stderr_log.close()

            if returncode == 0:
                print(f"Video saved successfully: {self.output_path}")
                print(f"Total frames: {self.frame_count}")
            else:
                print(f"FFmpeg encoding completed with return code {returncode}")
                # Print the log file if there was an error and logging is enabled
                if self.stderr_log_path:
                    try:
                        with open(self.stderr_log_path, 'r') as f:
                            log_contents = f.read()
                            if log_contents:
                                print(f"FFmpeg error log:\n{log_contents}")
                    except:
                        pass

        except subprocess.TimeoutExpired:
            print(f"FFmpeg did not finish within timeout, forcing termination...")
            self.ffmpeg.terminate()
            try:
                self.ffmpeg.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print(f"Force killing ffmpeg...")
                self.ffmpeg.kill()
                self.ffmpeg.wait()
            print(f"Video may be incomplete: {self.output_path}")

            # Close stderr log if it exists
            if self.stderr_log:
                self.stderr_log.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
