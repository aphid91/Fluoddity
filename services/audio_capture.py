"""Offline audio capture for video recording.

Realtime playback has to hit a deadline, so the tracer is paced by the audio
ring. Recording has no deadline: frames are produced as fast (or slow) as the
renderer manages, and the only thing that matters is that the finished file
holds the right number of samples.

So the offline path is driven per video frame instead. One frame of video is
1/fps of output and therefore needs exactly sample_rate/fps samples, whatever
speedmult is - speedmult changes how far the field evolves between samples,
not how many samples a frame is worth.

Samples are conditioned by the same shader, mix reduction, filters and
limiter as the realtime path; only the ring/fence/PortAudio delivery is
skipped. The result is written to a WAV and muxed into the video by ffmpeg.
"""
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np

from state.audio_state import AUDIO_BLOCK
from utilities.ffmpeg_recorder import find_ffmpeg


class AudioCapture:
    """Accumulates offline-rendered audio blocks and writes them to a WAV."""

    def __init__(self, sample_rate: int, video_fps: int):
        self.sample_rate = int(sample_rate)
        self.video_fps = max(1, int(video_fps))
        # Samples owed per video frame. Usually fractional (48000/60 = 800 is
        # exact, but 44100/60 = 735 and other rates are not), so the debt is
        # accumulated and paid in whole blocks.
        self.samples_per_frame = self.sample_rate / self.video_fps
        self._debt = 0.0
        self._chunks = []
        self._total = 0
        self.frames_captured = 0

    def samples_owed(self, frames: float = 1.0) -> int:
        """Whole samples to render for `frames` video frames.

        Fractional `frames` is expected: audio is generated per physics step,
        so each step owes 1/speedmult of a video frame. The debt accumulates,
        so the per-frame total stays exact however it is subdivided.

        Granularity is samples rather than whole blocks because a physics step
        owes far less than a block - 13 samples at speedmult=60, 32 at
        speedmult=25. Quantising to blocks meant a step that finally owed one
        rendered 512 samples, three to thirty-eight physics steps' worth of
        audio, against a single frozen canvas pair; the field then jumped
        forward all at once. That is the staircase the realtime path no longer
        has, and it is why recordings still zippered after realtime was fixed.
        """
        self._debt += self.samples_per_frame * frames
        self.frames_captured += frames
        n = max(0, int(self._debt))
        # Retire the debt here, not in add_samples(): every sample handed out
        # IS rendered by the caller, but they are only handed BACK once the
        # 512-sample block they landed in fills up. Waiting for that made the
        # debt monotonically grow, so each call re-requested the whole
        # outstanding backlog (53, 106, 160, 213 ...) and rendered it against
        # one frozen canvas - reintroducing the very staircase this replaced.
        self._debt -= n
        return n

    def add_samples(self, samples: np.ndarray):
        """Append rendered samples to the captured audio.

        Accepts any length, so a physics step can contribute its own small
        share rather than a whole block. Does not touch the debt: that is
        retired by samples_owed() when the work is dispatched.
        """
        self._chunks.append(np.asarray(samples, dtype=np.float32).copy())
        self._total += len(samples)

    @property
    def duration(self) -> float:
        return self._total / self.sample_rate if self.sample_rate else 0.0

    def has_audio(self) -> bool:
        return self._total > 0

    def write_wav(self, path: Path | str) -> Path | None:
        """Write the captured audio, trimmed to the exact video duration."""
        if not self._chunks:
            return None
        data = np.concatenate(self._chunks)

        # Blocks are whole, so the tail usually overshoots the last frame.
        # Trim to the exact number of samples the video is worth, otherwise
        # the audio would run slightly long and drift out of sync.
        wanted = int(round(self.frames_captured * self.samples_per_frame))
        if wanted > 0:
            if len(data) > wanted:
                data = data[:wanted]
            elif len(data) < wanted:
                data = np.pad(data, (0, wanted - len(data)))

        np.clip(data, -1.0, 1.0, out=data)
        pcm = (data * 32767.0).astype("<i2")

        path = Path(path)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return path


def mux_audio_into_video(video_path, wav_path, debug_log=False):
    """Remux a finished video with a WAV track, replacing the original file.

    Returns the output path on success, None on failure (leaving the silent
    video intact - losing the render because the audio failed would be worse
    than delivering it without sound).
    """
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    if not video_path.exists() or not wav_path.exists():
        return None

    ffmpeg = find_ffmpeg()
    merged = video_path.with_name(video_path.stem + "_audio" + video_path.suffix)
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(wav_path),
        "-c:v", "copy",          # No re-encode: the video is already final.
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(merged),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=(None if debug_log else subprocess.DEVNULL),
        )
    except Exception as e:
        print(f"Audio mux failed to launch ffmpeg: {e}")
        return None

    if proc.returncode != 0 or not merged.exists():
        print(f"Audio mux failed (ffmpeg exit {proc.returncode}); "
              f"keeping silent video: {video_path}")
        return None

    # Replace the silent render with the muxed one.
    try:
        video_path.unlink()
        merged.rename(video_path)
        wav_path.unlink(missing_ok=True)
        return video_path
    except OSError as e:
        print(f"Could not replace video with muxed version: {e}")
        return merged
