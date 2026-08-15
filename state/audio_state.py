from dataclasses import dataclass, field

# Marks a field as runtime-only, so it is skipped when preferences are saved.
TRANSIENT = {"transient": True}

# Audio block size, in frames. One tracer dispatch produces exactly this many
# samples, so STEPS_PER_DISPATCH is locked to it while audio is enabled.
AUDIO_BLOCK = 512

# GPU-side ring depth, in blocks. Absorbs GPU jitter between dispatch and
# readback.
#
# This has to be comfortably deeper than the per-frame block demand, because a
# fence issued during a frame has usually not signalled by the time that same
# frame's readback runs. At 48kHz/512 the stream needs 93.75 blocks/s, i.e.
# ~1.6 per 60fps frame; with only a handful of slots the pipeline stalls at
# ~1 block/frame and the audio runs slower than realtime.
AUDIO_SLOTS = 16

# CPU-side ring depth, in blocks. Must make the frame count a power of two.
AUDIO_RB_BLOCKS = 16


@dataclass
class AudioState:
    """Settings for the streamline-driven audio voice.

    First pass: a single voice driven by streamline index 0. The sample at
    each integration step is dot(velocity, field) * amplitude, high-passed to
    remove DC.
    """

    enabled: bool = False  # Master switch; drives the tracer clock when on
    show_window: bool = False  # Whether the Audio window is visible

    # Render the same voice into recorded video. Offline the audio is not
    # delivered on a deadline, so it is generated per video frame instead of
    # against the wall clock: one frame of video is 1/fps of output and needs
    # exactly sample_rate/fps samples, whatever speedmult is.
    record_audio: bool = False

    sample_rate: int = 48000  # Frames per second
    # Particles 0..voice_count-1 each contribute a voice to the mix. Kept
    # independent of the streamline count: sonifying a whole 1024-particle
    # swarm is mostly wash, and the reduction cost scales with this.
    voice_count: int = 1
    # 1/sqrt(n) is RMS-preserving for near-independent voices, so the level
    # holds steady as voice_count changes. 1/n would fade toward silence.
    rms_normalise: bool = True

    amplitude: float = 0.3  # Output gain applied to dot(vel, field)
    # dot(velocity, field) is unnormalised: velocity accumulates force every
    # step and routinely reaches ~100, so the raw product lands in the
    # hundreds. Auto-gain tracks the running peak and divides it out, keeping
    # Amplitude a usable 0..4 control instead of needing ~0.003.
    auto_gain: bool = True
    # Telemetry: gain the tracker is currently applying.
    auto_gain_db: float = field(default=0.0, metadata=TRANSIENT)
    highpass_hz: float = 20.0  # One-pole DC blocker cutoff
    limiter_ceiling: float = 0.9  # Peak ceiling before hard clip

    # Milliseconds of gain ramp applied after a hazard/edge reset. The
    # high-pass removes the DC step a reset causes, but the instantaneous
    # jump in dot() is still a click without this.
    reset_ramp_ms: float = 3.0

    # --- Read-only telemetry, updated by the service each frame ---
    # All transient: these describe the running stream, not user intent.
    starves: int = field(default=0, metadata=TRANSIENT)
    in_flight: int = field(default=0, metadata=TRANSIENT)
    ring_fill: float = field(default=0.0, metadata=TRANSIENT)
    peak: float = field(default=0.0, metadata=TRANSIENT)
    device_name: str = field(default="", metadata=TRANSIENT)
    last_error: str = field(default="", metadata=TRANSIENT)
