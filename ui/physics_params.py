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
        description="Scales axial, lateral, and sagittal forces applied to particles, and scales strafe power. Often tuned in the opposite direction to Sensor Gain and Drag to offset exploding/vanishing particle speed.",
    ),
    PhysicsParamDef(
        name='DRAG', label='Drag', group='forces',
        default_min=-1.0, default_max=1.0,
        hard_min=-1.0, hard_max=1.0,
        description="Each physics update, particle velocity is multiplied by drag like so:   vel = vel*drag + forces; So drag less than 1 means particles are being slowed down. Powerful (<0.5) drag values can prevent energetic systems from 'blowing up'",
    ),

    # --- Advanced group ---
    PhysicsParamDef(
        name='AXIAL_FORCE', label='Axial/Sagittal Force', group='advanced',
        default_min=-1.0, default_max=1.0,
        description="Controls the strength of forces applied parallel to the direction of travel and up/down: acceleration and braking, climbing and diving",
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


# === Lockable-parameter registry (Parameter Locks feature) ===
# The parameter_locks module derives its lockable-param list from here instead
# of hardcoding it, so it stays in sync with the params.
#
# The 12 physics sliders above are all lockable. In addition, several
# non-slider SimState settings are lockable. Those extras are not sliders, so
# they are NOT part of PHYSICS_PARAMS (which drives config_saver's sweep/jitter
# dicts and the physics window's slider rendering) — they are declared
# separately here.

# Extra lockable SimState params that are not physics sliders: (name, display label).
LOCKABLE_EXTRA_SIM_PARAMS: list[tuple[str, str]] = [
    # Additional settings
    ('rule_seed', 'Rule Seed'),
    ('boundary_conditions', 'Boundary Conditions'),
    ('initial_conditions', 'Initial Conditions'),
    ('init_spacing', 'Initialization Spacing'),
    ('num_cohorts', 'Number of Cohorts'),
    ('GRAVITY_FORCE', 'Gravity Force'),
    ('GRAVITY_STRAFE', 'Gravity Strafe'),
    ('DISABLE_SYMMETRY', 'Disable Symmetry'),
    ('ABSOLUTE_ORIENTATION', 'Absolute Orientation'),
    ('ORIENTATION_MIX', 'Orientation Mix'),
    # Appearance
    ('color_by_cohort', 'Color by Cohort'),
    ('hue_sensitivity', 'Hue Sensitivity'),
]

# Lockable PreferencesState params. (The force/strafe-field strengths that
# used to live here were removed with the drawing mode; the list is now empty
# but kept so the parameter_locks registry contract is unchanged.)
LOCKABLE_PREF_PARAMS: list[str] = []

# All lockable SimState param names, in a stable order: the 12 sliders first,
# then the extras. (Used to initialize the lock dict and to drive snapshot/restore.)
LOCKABLE_SIM_PARAMS: list[str] = (
    PHYSICS_PARAM_NAMES + [name for name, _label in LOCKABLE_EXTRA_SIM_PARAMS]
)

# name -> display label for every lockable SimState param (sliders + extras).
# Only slider params have entries in SimState.slider_ranges, but a label for
# every lockable param keeps the lock UI's [L] prefix consistent.
LOCKABLE_PARAM_LABELS: dict[str, str] = {
    **{p.name: p.label for p in PHYSICS_PARAMS},
    **{name: label for name, label in LOCKABLE_EXTRA_SIM_PARAMS},
}
