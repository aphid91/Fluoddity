from dataclasses import dataclass

# Upper bound on path length. The service allocates its buffer once at this
# size and clamps the step count to it, so changing steps never reallocates.
# Lives here rather than in the service so the UI can bound its slider without
# importing from services/ (which would add an edge to the ui/services cycle).
MAX_STEPS = 16384

# Upper bound on simultaneous streamlines. The path buffer holds
# MAX_STREAMLINES slices of (MAX_STEPS + 1) vec2.
MAX_STREAMLINES = 128


@dataclass
class StreamlineState:
    """Streamline probe settings.

    A streamline traces the path of a massless test particle released at the
    cursor, accelerated by the canvas vector field.
    """

    enabled: bool = False  # Whether the Streamlines window / overlay is active
    show_window: bool = False  # Whether the Streamlines control window is visible

    # Integration
    steps: int = 512  # Number of integration steps (path length in samples)
    force_scale: float = 1.0  # Canvas value -> acceleration multiplier
    damping: float = 0.98  # Per-step velocity retention (1.0 = frictionless)
    step_size: float = 0.01  # Velocity -> displacement multiplier per step
    # Spring pull back toward the seed. Scaled by step_size in the shader, so
    # useful values are large (hundreds+); 0 lets the particle drift free.
    restore_force: float = 0.0

    # Multi-streamline spray
    count: int = 1  # Number of streamlines launched from the seed
    initial_speed: float = 0.5  # Max magnitude of the random launch velocity
    resample_each_frame: bool = True  # Redraw launch directions every frame

    # Appearance
    color: tuple = (1.0, 0.85, 0.2)  # Line color (RGB)
    opacity: float = 0.9  # Line alpha
