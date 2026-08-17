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

# Upper bound on simultaneous voices, independent of MAX_STREAMLINES.
#
# The lane buffer is MAX_VOICES * AUDIO_SLOTS * AUDIO_BLOCK floats, so sizing
# it for all 8192 particles would reserve 537 MB for voices the mix shader
# cannot sum in realtime anyway - its inner loop is serial over voices, so it
# falls behind long before 8192. 1024 lanes cost 67 MB and are already more
# than the reduction keeps up with.
MAX_VOICES = 1024

# Furthest back the mix may read into a voice's history, in samples.
#
# The delay CANNOT read the production lanes. Those slots are reclaimed by
# pump() as soon as their fence signals and reused by the next dispatch, so in
# steady state the lane ring holds only the one or two blocks still in flight
# - about 21 ms, nowhere near a musical delay. Reading further back lands on
# samples a later dispatch already overwrote.
#
# So each voice gets a separate history ring that nothing reclaims: the tracer
# appends to it, and only wraparound ever overwrites. Its depth is what bounds
# the delay, independent of AUDIO_SLOTS.
#
# 16384 samples is 341 ms at 48 kHz. Power of two so the wrap is a mask.
MAX_VOICE_DELAY_SAMPLES = 16384


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

    # --- Per-voice delay ---
    # Each voice reads its lane a fixed distance in the past, the distance
    # drawn per-voice from [delay_min_ms, delay_max_ms] by a hash of the voice
    # index. Derived in the shader rather than stored, so it costs nothing at
    # 8192 voices and stays identical run to run.
    #
    # Note this is a smearing/chorus effect more than a decorrelator: the
    # voices were measured near-independent already (|corr| ~0.015 mean), so
    # there is little correlation for it to remove.
    voice_delay: bool = False
    delay_min_ms: float = 100.0
    delay_max_ms: float = 200.0

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
    # Blocks produced but discarded because the CPU ring was full. Non-zero
    # means the tracer is outrunning the sound device.
    overruns: int = field(default=0, metadata=TRANSIENT)
    # Outstanding GL fences. Should stay within [0, AUDIO_SLOTS]; a value that
    # climbs steadily means sync objects are leaking into the display driver,
    # which degrades the whole desktop rather than just this app.
    live_fences: int = field(default=0, metadata=TRANSIENT)
    in_flight: int = field(default=0, metadata=TRANSIENT)
    ring_fill: float = field(default=0.0, metadata=TRANSIENT)
    peak: float = field(default=0.0, metadata=TRANSIENT)
    device_name: str = field(default="", metadata=TRANSIENT)
    last_error: str = field(default="", metadata=TRANSIENT)
