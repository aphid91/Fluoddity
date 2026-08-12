"""
ConfigSaver service for saving/loading physics configurations.

Version 7: JSON format with all physics state (slider values, ranges, sweeps, appearance).
Legacy support for SIM1-6 binary formats for migration only.
"""
import base64
import zlib
import struct
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from state import SimState
from ui.physics_params import (
    PHYSICS_PARAMS as _PARAM_DEFS,
    PHYSICS_PARAM_NAMES as PHYSICS_PARAMS,
    DEFAULT_SLIDER_RANGES,
)

# Derived from single source of truth (ui/physics_params.py)
PARAM_TO_LABEL = {p.name: p.label for p in _PARAM_DEFS}

CONFIG_VERSION = 7
DEFAULT_RULE_SEED = 0.42

def _default_sweeps() -> dict[str, float]:
    """Create default sweep dict with all params set to 0.0."""
    return {p: 0.0 for p in PHYSICS_PARAMS}


def _default_jitters() -> dict[str, float]:
    """Create default jitters dict with all params set to 0.0."""
    return {p: 0.0 for p in PHYSICS_PARAMS}


def _default_slider_ranges() -> dict[str, list[float]]:
    """Create default slider ranges dict: [cur_min, cur_max, default_min, default_max]."""
    return {
        label: [r[0], r[1], r[0], r[1]]
        for label, r in DEFAULT_SLIDER_RANGES.items()
    }


@dataclass
class PhysicsConfig:
    """Complete physics configuration: all state from Physics Settings window."""

    # Physics parameters (12 sliders)
    axial_force: float = 0.371
    lateral_force: float = -0.707
    sensor_gain: float = 0.116
    mutation_scale: float = 0.0
    drag: float = 0.504
    strafe_power: float = 0.224
    sensor_angle: float = 0.45
    global_force_mult: float = 1.0
    sensor_distance: float = 1.0
    trail_persistence: float = 0.938
    trail_diffusion: float = 1.0
    hazard_rate: float = 0.0

    # Slider ranges: {label: [cur_min, cur_max, default_min, default_max]}
    slider_ranges: dict[str, list[float]] = field(default_factory=_default_slider_ranges)

    # Sweep states: {param_name: mode} where mode is 0.0, 1.0, or -1.0
    x_sweeps: dict[str, float] = field(default_factory=_default_sweeps)
    y_sweeps: dict[str, float] = field(default_factory=_default_sweeps)
    cohort_sweeps: dict[str, float] = field(default_factory=_default_sweeps)
    parameter_sweeps_enabled: bool = False

    # Jitter settings: {param_name: jitter_amount} where amount is 0.0-2.0
    jitters: dict[str, float] = field(default_factory=_default_jitters)

    # Simulation settings
    disable_symmetry: bool = False
    absolute_orientation: int = 0  # 0=Off, 1=Y axis, 2=Radial
    orientation_mix: float = 1.0
    boundary_conditions: int = 0  # 0=Bounce, 1=Reset, 2=Wrap
    initial_conditions: int = 0   # 0=Grid, 1=Random, 2=Ring
    num_cohorts: int = 64
    rule_seed: float = DEFAULT_RULE_SEED

    # Appearance settings
    ink_weight: float = 1.0
    hue_sensitivity: float = 0.5
    color_by_cohort: bool = True
    watercolor_mode: bool = False

    # Rule data (10 centers * 8 floats = 80 floats)
    rule: np.ndarray = field(default_factory=lambda: np.zeros((10, 8), dtype=np.float32))

    # User notes (optional)
    notes: str = ""

    # Field strength multipliers (only stored when field texture is present)
    force_field_strength: float | None = None
    strafe_field_strength: float | None = None

    def to_dict(self) -> dict:
        """Convert config to JSON-serializable dict."""
        d = {
            'version': CONFIG_VERSION,
            'physics': {
                'axial_force': self.axial_force,
                'lateral_force': self.lateral_force,
                'sensor_gain': self.sensor_gain,
                'mutation_scale': self.mutation_scale,
                'drag': self.drag,
                'strafe_power': self.strafe_power,
                'sensor_angle': self.sensor_angle,
                'global_force_mult': self.global_force_mult,
                'sensor_distance': self.sensor_distance,
                'trail_persistence': self.trail_persistence,
                'trail_diffusion': self.trail_diffusion,
                'hazard_rate': self.hazard_rate,
            },
            'slider_ranges': self.slider_ranges,
            'sweeps': {
                'x': self.x_sweeps,
                'y': self.y_sweeps,
                'cohort': self.cohort_sweeps,
            },
            'jitters': self.jitters,
            'parameter_sweeps_enabled': self.parameter_sweeps_enabled,
            'settings': {
                'disable_symmetry': self.disable_symmetry,
                'absolute_orientation': self.absolute_orientation,
                'orientation_mix': self.orientation_mix,
                'boundary_conditions': self.boundary_conditions,
                'initial_conditions': self.initial_conditions,
                'num_cohorts': self.num_cohorts,
                'rule_seed': self.rule_seed,
            },
            'appearance': {
                'ink_weight': self.ink_weight,
                'hue_sensitivity': self.hue_sensitivity,
                'color_by_cohort': self.color_by_cohort,
                'watercolor_mode': self.watercolor_mode,
            },
            'rule': self.rule.flatten().tolist(),
            'notes': self.notes,
        }
        if self.force_field_strength is not None:
            d['field_strengths'] = {
                'force': self.force_field_strength,
                'strafe': self.strafe_field_strength,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'PhysicsConfig':
        """Create config from JSON dict."""
        physics = data.get('physics', {})
        settings = data.get('settings', {})
        appearance = data.get('appearance', {})
        sweeps = data.get('sweeps', {})
        notes = data.get('notes', '')

        # Parse optional field strengths
        field_strengths = data.get('field_strengths')
        force_field_strength = field_strengths['force'] if field_strengths else None
        strafe_field_strength = field_strengths['strafe'] if field_strengths else None

        # Parse rule from flat list
        rule_list = data.get('rule', [0.0] * 80)
        rule = np.array(rule_list, dtype=np.float32).reshape(10, 8)

        # Ensure sweep dicts have all params (fill missing with 0.0)
        x_sweeps = _default_sweeps()
        x_sweeps.update(sweeps.get('x', {}))
        y_sweeps = _default_sweeps()
        y_sweeps.update(sweeps.get('y', {}))
        cohort_sweeps = _default_sweeps()
        cohort_sweeps.update(sweeps.get('cohort', {}))

        # Ensure jitters dict has all params (backwards compatibility: default 0.0)
        jitters = _default_jitters()
        jitters.update(data.get('jitters', {}))

        # Ensure slider_ranges has all sliders (fill missing with defaults)
        slider_ranges = _default_slider_ranges()
        slider_ranges.update(data.get('slider_ranges', {}))

        return cls(
            axial_force=physics.get('axial_force', 0.371),
            lateral_force=physics.get('lateral_force', -0.707),
            sensor_gain=physics.get('sensor_gain', 0.116),
            mutation_scale=physics.get('mutation_scale', 0.0),
            drag=physics.get('drag', 0.504),
            strafe_power=physics.get('strafe_power', 0.224),
            sensor_angle=physics.get('sensor_angle', 0.45),
            global_force_mult=physics.get('global_force_mult', 1.0),
            sensor_distance=physics.get('sensor_distance', 1.0),
            trail_persistence=physics.get('trail_persistence', 0.938),
            trail_diffusion=physics.get('trail_diffusion', 1.0),
            hazard_rate=physics.get('hazard_rate', 0.0),
            slider_ranges=slider_ranges,
            x_sweeps=x_sweeps,
            y_sweeps=y_sweeps,
            cohort_sweeps=cohort_sweeps,
            jitters=jitters,
            parameter_sweeps_enabled=data.get('parameter_sweeps_enabled', False),
            disable_symmetry=settings.get('disable_symmetry', False),
            absolute_orientation=int(settings.get('absolute_orientation', 0)),
            orientation_mix=settings.get('orientation_mix', 1.0),
            boundary_conditions=settings.get('boundary_conditions', 0),
            initial_conditions=settings.get('initial_conditions', 0),
            num_cohorts=settings.get('num_cohorts', 64),
            rule_seed=settings.get('rule_seed', DEFAULT_RULE_SEED),
            ink_weight=appearance.get('ink_weight', 1.0),
            hue_sensitivity=appearance.get('hue_sensitivity', 0.5),
            color_by_cohort=appearance.get('color_by_cohort', True),
            watercolor_mode=appearance.get('watercolor_mode', False),
            rule=rule,
            notes=notes,
            force_field_strength=force_field_strength,
            strafe_field_strength=strafe_field_strength,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'PhysicsConfig':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ConfigSaver:
    """Service for saving/loading physics configurations."""

    def create_config(self, sim_state: SimState, rule: np.ndarray | None,
                      field_strengths: tuple[float, float] | None = None) -> PhysicsConfig:
        """Create a PhysicsConfig from current state.

        Args:
            field_strengths: Optional (force_field_strength, strafe_field_strength) tuple.
                Only set when a non-zero field texture is being saved alongside.
        """
        if rule is None:
            rule = np.zeros((10, 8), dtype=np.float32)

        return PhysicsConfig(
            axial_force=sim_state.AXIAL_FORCE,
            lateral_force=sim_state.LATERAL_FORCE,
            sensor_gain=sim_state.SENSOR_GAIN,
            mutation_scale=sim_state.MUTATION_SCALE,
            drag=sim_state.DRAG,
            strafe_power=sim_state.STRAFE_POWER,
            sensor_angle=sim_state.SENSOR_ANGLE,
            global_force_mult=sim_state.GLOBAL_FORCE_MULT,
            sensor_distance=sim_state.SENSOR_DISTANCE,
            trail_persistence=sim_state.TRAIL_PERSISTENCE,
            trail_diffusion=sim_state.TRAIL_DIFFUSION,
            hazard_rate=sim_state.HAZARD_RATE,
            slider_ranges=sim_state.slider_ranges.copy(),
            x_sweeps=sim_state.x_sweeps.copy(),
            y_sweeps=sim_state.y_sweeps.copy(),
            cohort_sweeps=sim_state.cohort_sweeps.copy(),
            jitters=sim_state.jitters.copy(),
            parameter_sweeps_enabled=sim_state.parameter_sweeps_enabled,
            disable_symmetry=sim_state.DISABLE_SYMMETRY,
            absolute_orientation=sim_state.ABSOLUTE_ORIENTATION,
            orientation_mix=sim_state.ORIENTATION_MIX,
            boundary_conditions=sim_state.boundary_conditions,
            initial_conditions=sim_state.initial_conditions,
            num_cohorts=sim_state.num_cohorts,
            rule_seed=sim_state.rule_seed,
            ink_weight=sim_state.ink_weight,
            hue_sensitivity=sim_state.hue_sensitivity,
            color_by_cohort=sim_state.color_by_cohort,
            watercolor_mode=sim_state.watercolor_mode,
            rule=rule.copy(),
            notes=sim_state.notes,
            force_field_strength=field_strengths[0] if field_strengths else None,
            strafe_field_strength=field_strengths[1] if field_strengths else None,
        )

    def apply_config(self, config: PhysicsConfig, sim_state: SimState,
                     watercolor_override: bool | None = None) -> np.ndarray:
        """
        Apply a PhysicsConfig to the simulation state.

        Args:
            config: The config to apply
            sim_state: SimState to update (modified in place)
            watercolor_override: If not None, override config's watercolor_mode

        Returns:
            The rule to push to RuleManager
        """
        # Physics parameters
        sim_state.AXIAL_FORCE = config.axial_force
        sim_state.LATERAL_FORCE = config.lateral_force
        sim_state.SENSOR_GAIN = config.sensor_gain
        sim_state.MUTATION_SCALE = config.mutation_scale
        sim_state.DRAG = config.drag
        sim_state.STRAFE_POWER = config.strafe_power
        sim_state.SENSOR_ANGLE = config.sensor_angle
        sim_state.GLOBAL_FORCE_MULT = config.global_force_mult
        sim_state.SENSOR_DISTANCE = config.sensor_distance
        sim_state.TRAIL_PERSISTENCE = config.trail_persistence
        sim_state.TRAIL_DIFFUSION = config.trail_diffusion
        sim_state.HAZARD_RATE = config.hazard_rate

        # Slider ranges (full replacement)
        sim_state.slider_ranges.clear()
        sim_state.slider_ranges.update(config.slider_ranges)

        # Sweep states (full replacement)
        sim_state.x_sweeps.clear()
        sim_state.x_sweeps.update(config.x_sweeps)
        sim_state.y_sweeps.clear()
        sim_state.y_sweeps.update(config.y_sweeps)
        sim_state.cohort_sweeps.clear()
        sim_state.cohort_sweeps.update(config.cohort_sweeps)
        sim_state.parameter_sweeps_enabled = config.parameter_sweeps_enabled

        # Jitter states (full replacement)
        sim_state.jitters.clear()
        sim_state.jitters.update(config.jitters)

        # Simulation settings
        sim_state.DISABLE_SYMMETRY = config.disable_symmetry
        sim_state.ABSOLUTE_ORIENTATION = config.absolute_orientation
        sim_state.ORIENTATION_MIX = config.orientation_mix
        sim_state.boundary_conditions = config.boundary_conditions
        sim_state.initial_conditions = config.initial_conditions
        sim_state.num_cohorts = config.num_cohorts
        sim_state.rule_seed = config.rule_seed

        # Appearance settings
        sim_state.ink_weight = config.ink_weight
        sim_state.hue_sensitivity = config.hue_sensitivity
        sim_state.color_by_cohort = config.color_by_cohort
        sim_state.watercolor_mode = watercolor_override if watercolor_override is not None else config.watercolor_mode

        # User notes
        sim_state.notes = config.notes

        return config.rule.copy()

    # --- File I/O ---

    def save_to_file(self, config: PhysicsConfig, filepath: Path | str) -> None:
        """Save config to a JSON file."""
        filepath = Path(filepath)
        filepath.write_text(config.to_json())

    def load_from_file(self, filepath: Path | str) -> PhysicsConfig | None:
        """Load config from a file. Supports both JSON (v7+) and legacy SIM formats."""
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        content = filepath.read_text()

        # Check if it's a legacy SIM format
        if content.startswith('SIM'):
            return self.decode_legacy(content)

        # Otherwise parse as JSON
        try:
            return PhysicsConfig.from_json(content)
        except Exception as e:
            print(f"Failed to load config from {filepath}: {e}")
            return None

    # --- Clipboard (compressed JSON) ---

    def encode_clipboard(self, config: PhysicsConfig) -> str:
        """Encode config to compressed string for clipboard (Ctrl+C)."""
        json_bytes = config.to_json(indent=None).encode('utf-8')
        compressed = zlib.compress(json_bytes, level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        return f"SIM{CONFIG_VERSION}:{encoded}"

    def decode_clipboard(self, clipboard_str: str) -> PhysicsConfig | None:
        """Decode config from clipboard string (Ctrl+V). Supports legacy formats."""
        if isinstance(clipboard_str, bytes):
            clipboard_str = clipboard_str.decode('utf-8')

        if not clipboard_str.startswith('SIM'):
            print("Invalid config string: missing SIM prefix")
            return None

        try:
            colon_idx = clipboard_str.index(':')
            version = int(clipboard_str[3:colon_idx])
            encoded = clipboard_str[colon_idx + 1:]
            compressed = base64.urlsafe_b64decode(encoded)
            raw_bytes = zlib.decompress(compressed)

            if version == CONFIG_VERSION:
                # New JSON format
                return PhysicsConfig.from_json(raw_bytes.decode('utf-8'))
            elif version in [1, 2, 3, 4, 5, 6]:
                # Legacy binary format
                return self._from_legacy_bytes(raw_bytes, version)
            else:
                print(f"Unsupported config version: {version}")
                return None
        except Exception as e:
            print(f"Failed to decode clipboard: {e}")
            return None

    # --- Convenience methods ---

    def save_to_string(self, sim_state: SimState, rule: np.ndarray | None) -> str:
        """Create config and encode to clipboard string."""
        config = self.create_config(sim_state, rule)
        return self.encode_clipboard(config)

    def load_from_string(self, config_string: str, sim_state: SimState,
                         watercolor_override: bool | None = None) -> np.ndarray | None:
        """Decode string and apply to state. Returns rule or None if failed."""
        config = self.decode_clipboard(config_string)
        if config is None:
            return None
        return self.apply_config(config, sim_state, watercolor_override)

    # --- Legacy format support (for migration) ---

    def decode_legacy(self, config_string: str) -> PhysicsConfig | None:
        """Decode a legacy SIM1-6 format string."""
        return self.decode_clipboard(config_string)

    def _from_legacy_bytes(self, data: bytes, version: int) -> PhysicsConfig:
        """Convert legacy binary format to PhysicsConfig."""
        # Unpack physics params (10 floats = 40 bytes)
        physics = struct.unpack('10f', data[:40])
        # Unpack rule (80 floats = 320 bytes)
        rule_data = np.frombuffer(data[40:360], dtype=np.float32).copy()
        rule = rule_data.reshape(10, 8)

        # Default values
        disable_symmetry = False
        absolute_orientation = False
        boundary_conditions = 0
        initial_conditions = 0
        num_cohorts = 64
        rule_seed = DEFAULT_RULE_SEED
        ink_weight = 1.0
        hue_sensitivity = 0.5
        color_by_cohort = True
        watercolor_mode = False
        parameter_sweeps_enabled = False
        x_sweeps = _default_sweeps()
        y_sweeps = _default_sweeps()
        cohort_sweeps = _default_sweeps()

        # Version 2+: booleans
        if len(data) >= 362:
            disable_symmetry, absolute_orientation = struct.unpack('??', data[360:362])

        # Version 3+: simulation settings
        if len(data) >= 374:
            boundary_conditions, initial_conditions, num_cohorts = struct.unpack('3i', data[362:374])

        # Version 4+: rule_seed
        if len(data) >= 378:
            rule_seed, = struct.unpack('f', data[374:378])

        # Version 5+: appearance and sweeps
        if version >= 6 and len(data) >= 405:
            # Emboss fields still occupy their bytes on disk; read and discard them.
            _, ink_weight, hue_sensitivity, color_by_cohort, watercolor_mode, \
                _emboss_intensity, _emboss_smoothness, _emboss_mode = struct.unpack('<fff??ffi', data[378:404])
            parameter_sweeps_enabled, = struct.unpack('?', data[404:405])
            offset = 405
            x_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            y_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            cohort_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            # Convert old sweep format to new
            if x_sweep_data:
                x_sweeps[x_sweep_data[0]] = x_sweep_data[1]
            if y_sweep_data:
                y_sweeps[y_sweep_data[0]] = y_sweep_data[1]
            if cohort_sweep_data:
                cohort_sweeps[cohort_sweep_data[0]] = cohort_sweep_data[1]
        elif len(data) >= 401:
            # Emboss fields still occupy their bytes on disk; read and discard them.
            _, ink_weight, hue_sensitivity, color_by_cohort, watercolor_mode, \
                _emboss_intensity, _emboss_smoothness = struct.unpack('<fff??ff', data[378:400])
            parameter_sweeps_enabled, = struct.unpack('?', data[400:401])
            offset = 401
            x_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            y_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            cohort_sweep_data, offset = self._decode_legacy_sweep(data, offset)
            if x_sweep_data:
                x_sweeps[x_sweep_data[0]] = x_sweep_data[1]
            if y_sweep_data:
                y_sweeps[y_sweep_data[0]] = y_sweep_data[1]
            if cohort_sweep_data:
                cohort_sweeps[cohort_sweep_data[0]] = cohort_sweep_data[1]

        return PhysicsConfig(
            axial_force=physics[0],
            lateral_force=physics[1],
            sensor_gain=physics[2],
            mutation_scale=physics[3],
            drag=physics[4],
            strafe_power=physics[5],
            sensor_angle=physics[6],
            global_force_mult=physics[7],
            sensor_distance=physics[8],
            trail_persistence=physics[9],
            trail_diffusion=1.0,  # Legacy didn't have this param, use default
            slider_ranges=_default_slider_ranges(),  # Legacy didn't save all ranges
            x_sweeps=x_sweeps,
            y_sweeps=y_sweeps,
            cohort_sweeps=cohort_sweeps,
            parameter_sweeps_enabled=parameter_sweeps_enabled,
            disable_symmetry=disable_symmetry,
            absolute_orientation=absolute_orientation,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            num_cohorts=num_cohorts,
            rule_seed=rule_seed,
            ink_weight=ink_weight,
            hue_sensitivity=hue_sensitivity,
            color_by_cohort=color_by_cohort,
            watercolor_mode=watercolor_mode,
            rule=rule,
        )

    def _decode_legacy_sweep(self, data: bytes, offset: int) -> tuple[tuple | None, int]:
        """Decode a single sweep from legacy binary format."""
        name_len = struct.unpack('B', data[offset:offset + 1])[0]
        offset += 1
        if name_len == 0:
            return None, offset
        param_name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len
        direction, cur_min, cur_max = struct.unpack('3f', data[offset:offset + 12])
        offset += 12
        return (param_name, direction, cur_min, cur_max), offset
