from dataclasses import dataclass, field

from .audio_state import TRANSIENT

# Ring capacity per particle, in positions. The service allocates its buffers
# once at this size and clamps the tail slider to it, so nothing reallocates at
# runtime. Lives here rather than in the service so the UI can bound its
# sliders without importing from services/ (which would add an edge to the
# ui/services cycle).
MAX_STEPS = 4096

# Upper bound on simultaneous particles. The path buffer holds
# MAX_STREAMLINES slices of MAX_STEPS vec2 -> 8192 * 4096 * 8B = 268 MB.
#
# MAX_STEPS was cut from 16384 when this rose from 1024: the path buffer is
# allocated eagerly at the maximum, so 8192 particles at the old ring depth
# would have reserved 1.07 GB of VRAM at startup whether or not the count was
# ever turned up. 4096 ring entries is still 8x the default tail_length.
MAX_STREAMLINES = 8192

# Safety cap on catch-up dispatches per rendered frame. Without this, a hitch
# (or a paused debugger) would queue an unbounded burst of GPU work on resume.
# Audio mode needs a much higher ceiling: at 48kHz/512 the tracer owes ~1.6
# dispatches per 60fps frame, and dropping one is an audible gap rather than
# a dropped video frame, so audio raises this (see AUDIO_MAX_DISPATCHES).
MAX_DISPATCHES_PER_FRAME = 64
# One audio block may be split into several tracer dispatches so the field
# interpolation stays inside a single physics interval (see
# StreamlineService._field_split). A 512-sample block can split up to 16 ways,
# and audio needs ~1.6 blocks per 60fps frame, so this has to allow ~25.
AUDIO_MAX_DISPATCHES_PER_FRAME = 48


@dataclass
class StreamlineState:
    """Streamline probe settings.

    Particles are stateful: each dispatch advances them a few steps and appends
    to a per-particle ring buffer, so paths accumulate across dispatches rather
    than being retraced from the cursor every frame.
    """

    enabled: bool = False  # Whether the Streamlines window / overlay is active
    show_window: bool = False  # Whether the Streamlines control window is visible

    # Blend the field between the previous and current canvas frame instead of
    # sampling it as a staircase. Exposed as a toggle because it sits directly
    # in the audio path: if a buzz at the physics rate is coming from the
    # interpolation rather than from the field genuinely stepping, turning
    # this off is the fastest way to tell.
    field_interpolation: bool = True

    # --- Scheduling (independent of both render and physics cadence) ---
    dispatch_hz: float = 60.0  # Target tracer dispatches per second
    steps_per_dispatch: int = 8  # Integration steps advanced per dispatch
    running: bool = True  # Whether the tracer advances at all

    # --- Integration ---
    # Newtonian mode: the original tracer, which treats the canvas as a force
    # field and advects the particle along it. Much simpler than the default
    # read-only Fluoddity particle (two sensors + calculate_entity_behavior),
    # and traces the flow rather than reproducing particle behaviour. The four
    # settings below apply only in this mode.
    newtonian_mode: bool = False
    force_scale: float = 1.0  # Canvas value -> acceleration multiplier
    damping: float = 0.98  # Per-step velocity retention (1.0 = frictionless)
    step_size: float = 0.01  # Velocity -> displacement multiplier per step
    # Spring pull back toward the seed. Scaled by step_size in the shader, so
    # useful values are large (hundreds+); 0 lets the particle drift free.
    restore_force: float = 0.0

    # --- Population ---
    count: int = 1  # Number of particles
    initial_speed: float = 0.5  # Max magnitude of the random launch velocity
    # Radius of the disc each particle spawns into around the seed. At 0 they
    # all start from the same point and, running identical rules, trace
    # near-identical paths.
    seed_scatter: float = 0.12
    resample_each_frame: bool = True  # Redraw launch directions on each reset
    stop_at_edge: bool = True  # Retire particles that leave the canvas
    respawn_at_seed: bool = False  # Retired particles restart at the seed
    # Independent per-step probability that a particle respawns at the seed.
    # Per step (not per dispatch) so the rate is unaffected by the schedule.
    hazard_rate: float = 0.0

    # --- Seed pinning (middle-click toggles) ---
    seed_pinned: bool = False
    pinned_seed: tuple = (0.0, 0.0)  # World space [-1, 1]

    # --- Display ---
    render_to_video: bool = False  # Draw the overlay into recorded frames
    tail_length: int = 512  # Ring entries drawn per particle
    color: tuple = (1.0, 0.85, 0.2)  # Line color (RGB)
    opacity: float = 0.9  # Line alpha
    # Line thickness via glLineWidth. Core profile only guarantees 1.0, but
    # NVIDIA honours the full aliased range; the service clamps to whatever
    # the driver reports and silently falls back to 1px elsewhere.
    line_width: float = 1.0

    # --- One-shot flags (consumed by the service each frame) ---
    request_reset: bool = field(default=False, metadata=TRANSIENT)
