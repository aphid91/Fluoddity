from dataclasses import dataclass

# Ring capacity per particle, in positions. The service allocates its buffers
# once at this size and clamps the tail slider to it, so nothing reallocates at
# runtime. Lives here rather than in the service so the UI can bound its
# sliders without importing from services/ (which would add an edge to the
# ui/services cycle).
MAX_STEPS = 16384

# Upper bound on simultaneous particles. The path buffer holds
# MAX_STREAMLINES slices of MAX_STEPS vec2.
MAX_STREAMLINES = 128

# Safety cap on catch-up dispatches per rendered frame. Without this, a hitch
# (or a paused debugger) would queue an unbounded burst of GPU work on resume.
MAX_DISPATCHES_PER_FRAME = 64


@dataclass
class StreamlineState:
    """Streamline probe settings.

    Particles are stateful: each dispatch advances them a few steps and appends
    to a per-particle ring buffer, so paths accumulate across dispatches rather
    than being retraced from the cursor every frame.
    """

    enabled: bool = False  # Whether the Streamlines window / overlay is active
    show_window: bool = False  # Whether the Streamlines control window is visible

    # --- Scheduling (independent of both render and physics cadence) ---
    dispatch_hz: float = 60.0  # Target tracer dispatches per second
    steps_per_dispatch: int = 8  # Integration steps advanced per dispatch
    running: bool = True  # Whether the tracer advances at all

    # --- Integration ---
    force_scale: float = 1.0  # Canvas value -> acceleration multiplier
    damping: float = 0.98  # Per-step velocity retention (1.0 = frictionless)
    step_size: float = 0.01  # Velocity -> displacement multiplier per step
    # Spring pull back toward the seed. Scaled by step_size in the shader, so
    # useful values are large (hundreds+); 0 lets the particle drift free.
    restore_force: float = 0.0

    # --- Population ---
    count: int = 1  # Number of particles
    initial_speed: float = 0.5  # Max magnitude of the random launch velocity
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
    tail_length: int = 512  # Ring entries drawn per particle
    color: tuple = (1.0, 0.85, 0.2)  # Line color (RGB)
    opacity: float = 0.9  # Line alpha
    # Line thickness via glLineWidth. Core profile only guarantees 1.0, but
    # NVIDIA honours the full aliased range; the service clamps to whatever
    # the driver reports and silently falls back to 1px elsewhere.
    line_width: float = 1.0

    # --- One-shot flags (consumed by the service each frame) ---
    request_reset: bool = False  # Re-seed every particle at the seed position
