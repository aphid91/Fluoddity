from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from utilities.paths import get_user_preferences_path


@dataclass
class PreferencesState:
    """User preferences that persist between program sessions."""

    # Camera/rendering preferences
    speedmult: int = 5
    motion_blur: bool = True
    blur_quality: int = 2  # Motion blur render cadence (1 = every frame, 2 = every 2 frames, etc.)
    world_size: float = 0.40  # World size multiplier (affects entity count and canvas dimensions)
    canvas_aspect_ratio: str = "1:1"  # Canvas aspect ratio (e.g. "1:1", "16:9", "3:4")
    rule_seed: float = 0.0
    brightness: float = 3.0  # Global brightness multiplier
    tonemap_softness: float = 2.5  # Asinh tonemap stretch (higher = more highlight compression)
    exposure: float = 0.0  # Frame blending for motion blur effect (0=disabled, 1=long exposure)
    bloom_enabled: bool = True  # Whether bloom post-processing is active
    bloom_threshold: float = 0.11  # Brightness threshold for bloom extraction
    bloom_intensity: float = 0.23  # Bloom contribution strength
    bloom_radius: float = 1.0  # Bloom blur spread

    # UI preferences
    show_preferences_window: bool = True  # Whether preferences window is visible
    show_controls_window: bool = False  # Help controls window
    show_parameter_sweeps_window: bool = False  # Help parameter sweeps window
    show_tutorial_window: bool = True  # Help tutorial window
    show_performance_window: bool = False  # Help performance window
    physics_tooltips_enabled: bool = True
    debug_arrows: bool = False  # Visual debug overlay for velocity field
    arrow_sensitivity: float = 15.0  # Velocity scale for debug arrows (pow(2, x))
    mouse_mode: str = "Select Particle"  # "Select Particle" or "Draw Trail"
    draw_size: float = 0.031  # Gaussian kernel width for trail drawing
    draw_power: float = 1.0  # Velocity strength when drawing trails
    menu_close_threshold: float = 80.0  # Distance in pixels before menus auto-close

    # Advanced Drawing preferences
    advanced_drawing_enabled: bool = False  # Whether Advanced Drawing window is shown
    advanced_draw_canvas: bool = True  # "Trails / Canvas (Default)" checkbox
    advanced_draw_force_field: bool = False  # "Force Field" checkbox
    advanced_draw_strafe_field: bool = False  # "Strafe Field" checkbox
    brush_mode: int = 0  # 0=Mouse Direction, 1=Inverse, 2=Fixed, 3=Attract, 4=Repel
    fixed_direction_heading: float = 0.0  # Range -PI to PI, heading for Fixed Direction mode
    force_field_strength: float = 1.0  # Multiplier for force field effect (log scale 0.0001-10.0)
    strafe_field_strength: float = 1.0  # Multiplier for strafe field effect (log scale 0.0001-10.0)
    draw_target_overlay_opacity: float = 0.0  # Opacity of draw target field overlay in frame assembly (0-1)
    shader_driven_field: bool = False  # Use a frag shader to override the field texture
    field_override_shader: str = "march.frag"  # Currently selected field override shader filename

    # Physics slider group collapsed states (True = expanded/open, False = collapsed)
    physics_group_basics: bool = True  # Default: open (trail sensors + mutation)
    physics_group_forces: bool = True  # Default: open (global force mult, drag)
    physics_group_advanced: bool = False  # Default: collapsed
    physics_group_additional: bool = False  # Default: collapsed
    physics_group_notes: bool = False  # Default: collapsed

    # Load menu collapsed states (True = expanded/open, False = collapsed)
    load_menu_core_open: bool = True  # Default: open
    load_menu_custom_open: bool = True  # Default: open
    load_menu_advanced_open: bool = False  # Default: collapsed

    # Recording preferences
    max_frames: int = 1800  # 150 * 12
    motion_blur_samples: int = 12
    supersample_k: int = 1
    filename_prefix: str = ""
    recording_motion_blur: bool = True  # Motion blur setting used during video recording
    recording_blur_quality: int = 1  # Blur quality setting used during video recording
    video_end_frame: int = 0  # Target frame for video to end on (0 = disabled, start immediately)

    # Simulation determinism
    strong_determinism: bool = False  # Double-buffer canvas for fully deterministic simulation

    # Parameter locks
    parameter_locks_enabled: bool = False  # Master toggle for parameter lock feature

    # Generics (live-coding scratch uniforms)
    show_generics_window: bool = False
    generic0: float = 0.0
    generic1: float = 0.0
    generic2: float = 0.0
    generic3: float = 0.0
    generic4: float = 0.0
    generic5: float = 0.0
    generic6: float = 0.0
    generic7: float = 0.0

    # Plotting (GPU histogram reports)
    show_plotting_window: bool = False

    # Lottery-based natural selection (persistent evolving rules)
    lottery_enabled: bool = False


def save_preferences(prefs: PreferencesState, filepath: Path | str = None) -> None:
    """Save preferences to a JSON file."""
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(prefs)
    filepath.write_text(json.dumps(data, indent=2))


def load_preferences(filepath: Path | str = None) -> PreferencesState:
    """Load preferences from a JSON file. Returns default preferences if file doesn't exist."""
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    if not filepath.exists():
        return PreferencesState()

    try:
        data = json.loads(filepath.read_text())
        # Filter out any fields that are no longer in PreferencesState (backward compat)
        valid_fields = set(PreferencesState.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return PreferencesState(**filtered_data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Failed to load preferences from {filepath}: {e}")
        print("Using default preferences")
        return PreferencesState()
