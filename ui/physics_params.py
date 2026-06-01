"""Single source of truth for physics parameter definitions.

Every physics slider's metadata lives here: name, label, group, ranges,
tooltip, and special flags. Other modules (config_saver, slider_widgets,
physics_window, sim.py) import from here instead of maintaining their own
duplicate copies.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsParamDef:
    """Definition for a single physics slider parameter."""
    name: str           # SimState field name, e.g. 'SENSOR_GAIN'
    label: str          # Display label, e.g. 'Sensor Gain'
    group: str          # Collapsing header group: 'basics', 'forces', 'advanced'
    default_min: float  # Default slider minimum
    default_max: float  # Default slider maximum
    description: str    # Tooltip text

    # Optional hard limits (None = no hard limit)
    hard_min: float | None = None
    hard_max: float | None = None

    # Special flags
    hide_jitter: bool = False       # Hide jitter control in context menu
    is_power_scaled: bool = False   # Use power-scaled slider (Hazard Rate)
    power_exponent: float = 3.0     # Exponent for power scaling

    @property
    def config_attr(self) -> str:
        """PhysicsConfig attribute name (lowercase), e.g. 'sensor_gain'."""
        return self.name.lower()

    @property
    def uniform_name(self) -> str:
        """GLSL uniform struct name, e.g. 'SENSOR_GAIN_SETTING'."""
        return f'{self.name}_SETTING'


# === The Registry ===
# Ordered list of all physics slider parameters.
# This is the SINGLE SOURCE OF TRUTH — all other modules derive from this.

PHYSICS_PARAMS: list[PhysicsParamDef] = [
    # --- Basics group ---
    PhysicsParamDef(
        name='SENSOR_GAIN', label='Sensor Gain', group='basics',
        default_min=0.0, default_max=10.0,
        description="Determines how strongly particles respond to sensor input. Higher values make particles more reactive to the trails they sense on the Canvas.",
    ),
    PhysicsParamDef(
        name='SENSOR_ANGLE', label='Sensor Angle', group='basics',
        default_min=-1.0, default_max=1.0,
        hard_min=-1.0, hard_max=1.0,
        description="Sets the angular offset of particle sensors from their forward direction. Determines whether particles are 'looking ahead' or 'looking behind'.",
    ),
    PhysicsParamDef(
        name='SENSOR_DISTANCE', label='Sensor Distance', group='basics',
        default_min=0.0, default_max=3.0,
        description="Determines distance between a particle's center and where it reads the trail information from Canvas. Longer distances tend to create larger scale patterns.",
    ),
    PhysicsParamDef(
        name='MUTATION_SCALE', label='Mutation Scale', group='basics',
        default_min=0.0, default_max=1.0,
        hide_jitter=True,
        description="Controls the size of the random mutations applied to a rule when a new particle is clicked. At 0, every particle will behave exactly like the selected particle.",
    ),

    # --- Forces group ---
    PhysicsParamDef(
        name='GLOBAL_FORCE_MULT', label='Global Force Mult', group='forces',
        default_min=0.0, default_max=2.0,
        description="Scales axial and lateral forces applied to particles, and scales strafe power. Often tuned in the opposite direction to Sensor Gain and Drag to offset exploding/vanishing particle speed.",
    ),
    PhysicsParamDef(
        name='DRAG', label='Drag', group='forces',
        default_min=-1.0, default_max=1.0,
        hard_min=-1.0, hard_max=1.0,
        description="Each physics update, particle velocity is multiplied by drag like so:   vel = vel*drag + forces; So drag less than 1 means particles are being slowed down. Powerful (<0.5) drag values can prevent energetic systems from 'blowing up'",
    ),

    # --- Advanced group ---
    PhysicsParamDef(
        name='AXIAL_FORCE', label='Axial Force', group='advanced',
        default_min=-1.0, default_max=1.0,
        description="Controls the strength of forces applied parallel to the direction of travel: acceleration and braking",
    ),
    PhysicsParamDef(
        name='LATERAL_FORCE', label='Lateral Force', group='advanced',
        default_min=-1.0, default_max=1.0,
        description="Controls the strength of forces applied perpendicular to the direction of travel: turning left and right.",
    ),
    PhysicsParamDef(
        name='STRAFE_POWER', label='Strafe Power', group='advanced',
        default_min=0.0, default_max=0.5,
        description="Controls particle movement without applying forces to velocity. 'Strafe' is a vector added directly to position each frame, like a little hop. Strafe power scales with Axial, Lateral, and Global force multipliers.",
    ),
    PhysicsParamDef(
        name='TRAIL_PERSISTENCE', label='Trail Persistence', group='advanced',
        default_min=0.0, default_max=1.0,
        hard_min=0.0, hard_max=1.0,
        description="Controls how long particle trails remain visible. Higher values create longer-lasting trails, lower values make trails fade quickly. Values close to 1.0 tend to create 'sharper' more stable patterns. ",
    ),
    PhysicsParamDef(
        name='TRAIL_DIFFUSION', label='Trail Diffusion', group='advanced',
        default_min=0.0, default_max=1.0,
        hard_min=0.0, hard_max=1.0,
        description="Controls how quickly particle trails spread out and blend together.",
    ),
    PhysicsParamDef(
        name='ADVECTION', label='Advection', group='advanced',
        default_min=0.0, default_max=5.0,
        description="Advects the trail field along its own stored flow (semi-Lagrangian). Before diffusion/fade, the canvas is sampled one step upstream of the local current in canvas.xy, so trails drift along their own flow instead of only diffusing isotropically. 0 = off. (Negative values, reachable by widening the range, run it backward / sharpen against the flow.)",
    ),
    PhysicsParamDef(
        name='ADAPTIVE_SENSING', label='Adaptive Sensing', group='advanced',
        default_min=0.0, default_max=3.0,
        description="Gives each particle a memory of recent trail exposure that modulates its sensing reach: particles that have been sitting in dense trail desensitize/contract and resensitize as they move on, producing breathing/wave dynamics. Scales the (symmetry-preserving) sensor distance only, never the turn. 0 = off.",
    ),
    PhysicsParamDef(
        name='HAZARD_RATE', label='Hazard Rate', group='advanced',
        default_min=0.0, default_max=0.05,
        hard_min=0.0, hard_max=0.05,
        hide_jitter=True,
        is_power_scaled=True, power_exponent=3.0,
        description="Probability per frame that particles reset to initial conditions. Gives particles a probabalistic 'lifetime' after which they reset.",
    ),
]


# === Derived lookup helpers ===
# Built once at import time from the registry above.

PARAM_BY_NAME: dict[str, PhysicsParamDef] = {p.name: p for p in PHYSICS_PARAMS}
PARAM_BY_LABEL: dict[str, PhysicsParamDef] = {p.label: p for p in PHYSICS_PARAMS}

# Ordered list of just the parameter names (used by SimState defaults, config_saver, etc.)
PHYSICS_PARAM_NAMES: list[str] = [p.name for p in PHYSICS_PARAMS]

# Groups in display order
PARAM_GROUPS: dict[str, list[PhysicsParamDef]] = {}
for _p in PHYSICS_PARAMS:
    PARAM_GROUPS.setdefault(_p.group, []).append(_p)

# Default slider ranges keyed by label: {label: [default_min, default_max]}
DEFAULT_SLIDER_RANGES: dict[str, list[float]] = {
    p.label: [p.default_min, p.default_max] for p in PHYSICS_PARAMS
}
