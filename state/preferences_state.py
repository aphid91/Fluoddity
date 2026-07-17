from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from utilities.paths import get_user_preferences_path


# ---------------------------------------------------------------------------
# Per-module preference slices
#
# PreferencesState is composed of these slices, one per subsystem, so each
# module owns its own preferences. Nested access is used everywhere:
#   prefs.tracer.sdf_enabled, prefs.optix.ao_radius, prefs.rendering.speedmult
#
# Backward compat with pre-slice (flat-key) JSON files is handled by
# _FLAT_KEY_MAP + from_flat_dict() in load_preferences().
# ---------------------------------------------------------------------------


@dataclass
class RenderingPrefs:
    """Core frame rendering / motion-blur / tonemap preferences."""
    renderer: int = 0  # Active 3D renderer: 0=OpenGL (GL points / volumetric tracer), 1=Optix
    speedmult: int = 5
    motion_blur: bool = True
    blur_quality: int = 2  # Motion blur render cadence (1 = every frame, 2 = every 2 frames, etc.)
    world_size: float = 0.40  # Legacy (unused, kept for saved-prefs backward compat)
    canvas_aspect_ratio: str = "1:1"  # Legacy (unused, kept for saved-prefs backward compat)
    entity_count: int = 4000000  # Number of active particles (all allocated entities are active)
    canvas_resolution: int = 256  # Cubic canvas dimension (W=H=D) for 3D trail textures
    rule_seed: float = 0.0
    brightness: float = 3.0  # Global brightness multiplier
    tonemap_softness: float = 2.5  # Asinh tonemap stretch (higher = more highlight compression)

    # Shared Pathtrace controls — synced between both renderers (like Camera).
    rt_mode: int = 0  # Pathtrace mode: 0=Off (OptiX rasterize), 1=X spp, 2=Accumulate
    rt_samples: int = 1  # Samples/frame in "X spp" mode (OptiX only; hidden for OpenGL)
    capture_spp: int = 64  # Target SPP for a Re-render Preview / offline capture
    render_resolution_scale: float = 1.0  # Multiplier on render resolution
    firefly_clamp: bool = True  # Clamp per-sample radiance to kill fireflies
    firefly_clamp_max: float = 50.0  # Max luminance per sample when clamping


@dataclass
class BloomPrefs:
    """Bloom post-processing preferences."""
    enabled: bool = True  # Whether bloom post-processing is active
    threshold: float = 0.11  # Brightness threshold for bloom extraction
    intensity: float = 0.23  # Bloom contribution strength
    radius: float = 1.0  # Bloom blur spread


@dataclass
class RecordingPrefs:
    """Video recording preferences."""
    max_frames: int = 1800  # 150 * 12
    motion_blur_samples: int = 12
    supersample_k: int = 1
    filename_prefix: str = ""
    recording_motion_blur: bool = True  # Motion blur setting used during video recording
    recording_blur_quality: int = 1  # Blur quality setting used during video recording
    video_end_frame: int = 0  # Target frame for video to end on (0 = disabled, start immediately)


@dataclass
class GenericsPrefs:
    """Live-coding scratch uniform values."""
    generic0: float = 0.0
    generic1: float = 0.0
    generic2: float = 0.0
    generic3: float = 0.0
    generic4: float = 0.0
    generic5: float = 0.0
    generic6: float = 0.0
    generic7: float = 0.0


@dataclass
class ParameterLocksPrefs:
    """Parameter lock feature preferences."""
    enabled: bool = False  # Master toggle for parameter lock feature


@dataclass
class TracerPrefs:
    """Volumetric path tracer preferences."""
    sdf_enabled: bool = True
    colored_extinction: bool = False
    extinction_rgb: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    albedo_saturation: float = 1.0
    albedo_brightness: float = 0.8
    density_scale: float = 0.0001
    hg_g: float = 0.0  # HG phase asymmetry [-1,1]
    emission_strength: float = 0.0  # emission intensity (0 = off)
    # Sun/sky lighting moved to the shared LightingPrefs slice.
    # rt mode / capture spp / resolution scale / firefly clamp moved to the
    # shared RenderingPrefs slice (synced with OptiX). Tonemapping now uses the
    # shared Brightness / Tonemap Softness curve (RenderingPrefs), so the old
    # per-tracer `exposure` knob was removed.
    max_bounces: int = 0  # 0=unbounded (RR only)
    density_resolution_log2: int = 9   # 2^9 = 512
    color_resolution_log2: int = 9     # 2^9 = 512
    majorant_resolution_log2: int = 7  # 2^7 = 128


@dataclass
class LightingPrefs:
    """Shared sun + sky lighting, consumed by BOTH renderers (OpenGL volumetric
    tracer and OptiX path tracer) so their lighting stays unified. Uses the
    OptiX terminology/formatting. (OptiX-only extras like the cos-lobe env sky
    stay on OptixPrefs.)"""
    light_direction: list = field(default_factory=lambda: [0.577, 0.577, 0.577])  # unit, scene -> sun
    light_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    light_intensity: float = 1.0
    nee: bool = True  # Next Event Estimation (sun shadow rays)
    photosphere: bool = False  # equirectangular skybox for the sky
    sky_color_top: list = field(default_factory=lambda: [0.45, 0.62, 0.85])  # zenith
    sky_color_bottom: list = field(default_factory=lambda: [0.08, 0.08, 0.10])  # nadir
    sky_intensity: float = 1.0  # (volumetric tracer only; OptiX gradient sky is unscaled)


@dataclass
class OptixPrefs:
    """OptiX raytracer + path tracer preferences (spheres and PT are one subsystem).

    Sun/sky lighting moved to the shared LightingPrefs slice; only OptiX-specific
    controls remain here."""
    # OptiX raytracer settings
    enabled: bool = False
    sphere_radius_scale: float = 1.0
    shadows_enabled: bool = True
    ambient: float = 0.12
    ao_enabled: bool = False
    ao_num_rays: int = 2
    ao_radius: float = 0.5
    ambient_color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])  # Rasterize ambient tint (scaled by `ambient`)
    rz_depth_of_field: bool = False  # Use thin-lens DOF in rasterize mode (nearly free with denoiser)
    albedo_saturation: float = 0.8
    albedo_brightness: float = 1.0
    sphere_size_jitter: float = 0.0  # Per-sphere radius jitter to reduce banding (0-1)
    use_curves: bool = False  # Toggle: render entities as round linear curves instead of spheres
    curve_length: float = 1.0  # Distance between curve control points (multiplier on entity_size * radius_scale)
    curve_r0: float = 1.0  # Radius at first control point (multiplier on entity_size * radius_scale)
    curve_r1: float = 0.5  # Radius at second control point (multiplier on entity_size * radius_scale)
    sdf_enabled: bool = False  # Enable SDF scene geometry in OptiX renderers
    # (resolution_scale, rt_mode, rt_realtime_samples, rt_preview_spp,
    #  pt_firefly_clamp[_max] moved to shared RenderingPrefs — synced across renderers.)

    # OptiX RT mode and path tracer settings
    rz_samples: int = 1  # Samples/frame for Rasterize (Off) mode (1-8) — OptiX only
    pt_max_bounces: int = 8
    pt_rr_start_depth: int = 3
    pt_global_material: int = 0  # 0=Lambert, 1=Glossy, 2=Mirror
    pt_glossy_ior: float = 1.5
    pt_emission_intensity: float = 10.0  # Emissive radiance multiplier for negative-hue entities
    pt_denoise_enabled: bool = False
    rz_denoise_enabled: bool = False  # Denoise in rasterize mode (separate beauty pass)
    pt_env_sky_nee: bool = False  # Use cosine-lobe environment sky for NEE instead of directional sun
    pt_sun_exp: float = 15.0  # "Sun Sharpness": cos-lobe sky sun exponent (1..256), only used when pt_env_sky_nee


@dataclass
class Camera3DPrefs:
    """3D camera preferences."""
    fov: float = 50.0
    aperture: float = 0.0  # DOF lens radius
    focal_plane_depth: float = 5.0  # DOF focal plane distance
    move_speed: float = 2.0
    rotate_speed: float = 2.0
    orbit_center: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orbit_rate: float = 0.0
    stereogram: bool = False
    eye_offset: float = 0.1
    stereo_toe_in: bool = False
    stereo_wall_eye: bool = False


@dataclass
class UIWindowsPrefs:
    """Window visibility flags + general UI interaction preferences."""
    # Window visibility
    show_preferences_window: bool = True
    show_controls_window: bool = False
    show_parameter_sweeps_window: bool = False
    show_tutorial_window: bool = True
    show_performance_window: bool = False
    show_generics_window: bool = False
    show_radio_window: bool = False
    show_plotting_window: bool = False
    show_video_recording_window: bool = False
    show_scheduled_renders_window: bool = False
    show_render_settings_window: bool = False  # unified per-renderer controls
    show_config_clipboard_window: bool = False

    # UI interaction
    physics_tooltips_enabled: bool = True
    mouse_mode: str = "Select Particle"  # Mouse interaction mode (currently only "Select Particle")
    menu_close_threshold: float = 80.0  # Distance in pixels before menus auto-close

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

    # Render-settings window collapsing-header open states (True = expanded/open)
    render_group_camera: bool = True
    render_group_lighting: bool = True
    render_group_sky: bool = True
    render_group_geometry: bool = True
    render_group_material: bool = True
    render_group_rasterize: bool = True
    render_group_pathtrace: bool = True
    render_group_postprocess: bool = True
    render_group_medium: bool = True
    render_group_grid_resolutions: bool = False
    render_group_opengl_postprocess: bool = True


@dataclass
class PreferencesState:
    """User preferences that persist between program sessions.

    Composed of per-module slices so each subsystem owns its own prefs.
    Access is nested: prefs.rendering.speedmult, prefs.tracer.sdf_enabled, etc.
    """
    rendering: RenderingPrefs = field(default_factory=RenderingPrefs)
    bloom: BloomPrefs = field(default_factory=BloomPrefs)
    recording: RecordingPrefs = field(default_factory=RecordingPrefs)
    generics: GenericsPrefs = field(default_factory=GenericsPrefs)
    parameter_locks: ParameterLocksPrefs = field(default_factory=ParameterLocksPrefs)
    lighting: LightingPrefs = field(default_factory=LightingPrefs)
    tracer: TracerPrefs = field(default_factory=TracerPrefs)
    optix: OptixPrefs = field(default_factory=OptixPrefs)
    camera3d: Camera3DPrefs = field(default_factory=Camera3DPrefs)
    ui_windows: UIWindowsPrefs = field(default_factory=UIWindowsPrefs)


# The slice attribute names on PreferencesState (top-level nested-JSON keys).
_SLICE_ATTRS = tuple(PreferencesState.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Flat-key compatibility map: old flat JSON key -> (slice_attr, nested_field).
# The single source of truth for legacy-JSON migration and for the flat
# snapshot/restore used by services/render_spec.py.
# ---------------------------------------------------------------------------
_FLAT_KEY_MAP: dict[str, tuple[str, str]] = {
    # RenderingPrefs
    "renderer": ("rendering", "renderer"),
    "speedmult": ("rendering", "speedmult"),
    "motion_blur": ("rendering", "motion_blur"),
    "blur_quality": ("rendering", "blur_quality"),
    "world_size": ("rendering", "world_size"),
    "canvas_aspect_ratio": ("rendering", "canvas_aspect_ratio"),
    "entity_count": ("rendering", "entity_count"),
    "canvas_resolution": ("rendering", "canvas_resolution"),
    "rule_seed": ("rendering", "rule_seed"),
    "brightness": ("rendering", "brightness"),
    "tonemap_softness": ("rendering", "tonemap_softness"),
    # Shared Pathtrace controls (canonical flat keys = old OptiX names, so old
    # prefs migrate their OptiX values into the shared fields).
    "three_d_rt_mode": ("rendering", "rt_mode"),
    "three_d_rt_realtime_samples": ("rendering", "rt_samples"),
    "three_d_rt_preview_spp": ("rendering", "capture_spp"),
    "three_d_optix_resolution_scale": ("rendering", "render_resolution_scale"),
    "three_d_pt_firefly_clamp": ("rendering", "firefly_clamp"),
    "three_d_pt_firefly_clamp_max": ("rendering", "firefly_clamp_max"),
    # BloomPrefs
    "bloom_enabled": ("bloom", "enabled"),
    "bloom_threshold": ("bloom", "threshold"),
    "bloom_intensity": ("bloom", "intensity"),
    "bloom_radius": ("bloom", "radius"),
    # RecordingPrefs
    "max_frames": ("recording", "max_frames"),
    "motion_blur_samples": ("recording", "motion_blur_samples"),
    "supersample_k": ("recording", "supersample_k"),
    "filename_prefix": ("recording", "filename_prefix"),
    "recording_motion_blur": ("recording", "recording_motion_blur"),
    "recording_blur_quality": ("recording", "recording_blur_quality"),
    "video_end_frame": ("recording", "video_end_frame"),
    # GenericsPrefs
    "generic0": ("generics", "generic0"),
    "generic1": ("generics", "generic1"),
    "generic2": ("generics", "generic2"),
    "generic3": ("generics", "generic3"),
    "generic4": ("generics", "generic4"),
    "generic5": ("generics", "generic5"),
    "generic6": ("generics", "generic6"),
    "generic7": ("generics", "generic7"),
    # ParameterLocksPrefs
    "parameter_locks_enabled": ("parameter_locks", "enabled"),
    # TracerPrefs
    "tracer_sdf_enabled": ("tracer", "sdf_enabled"),
    "tracer_colored_extinction": ("tracer", "colored_extinction"),
    "tracer_extinction_rgb": ("tracer", "extinction_rgb"),
    "tracer_albedo_saturation": ("tracer", "albedo_saturation"),
    "tracer_albedo_brightness": ("tracer", "albedo_brightness"),
    "tracer_density_scale": ("tracer", "density_scale"),
    "tracer_hg_g": ("tracer", "hg_g"),
    "tracer_emission_strength": ("tracer", "emission_strength"),
    "tracer_max_bounces": ("tracer", "max_bounces"),
    "tracer_density_resolution_log2": ("tracer", "density_resolution_log2"),
    "tracer_color_resolution_log2": ("tracer", "color_resolution_log2"),
    "tracer_majorant_resolution_log2": ("tracer", "majorant_resolution_log2"),
    # LightingPrefs (shared sun + sky, used by both renderers).
    # Canonical flat keys use the OptiX names for backward-compat migration.
    "three_d_optix_light_direction": ("lighting", "light_direction"),
    "three_d_optix_light_color": ("lighting", "light_color"),
    "three_d_optix_light_intensity": ("lighting", "light_intensity"),
    "three_d_pt_sun_sampling": ("lighting", "nee"),
    "three_d_pt_photosphere": ("lighting", "photosphere"),
    "three_d_optix_sky_color_top": ("lighting", "sky_color_top"),
    "three_d_optix_sky_color_bottom": ("lighting", "sky_color_bottom"),
    "lighting_sky_intensity": ("lighting", "sky_intensity"),
    # OptixPrefs (spheres)
    "three_d_optix_enabled": ("optix", "enabled"),
    "three_d_optix_sphere_radius_scale": ("optix", "sphere_radius_scale"),
    "three_d_optix_shadows_enabled": ("optix", "shadows_enabled"),
    "three_d_optix_ambient": ("optix", "ambient"),
    "three_d_optix_ao_enabled": ("optix", "ao_enabled"),
    "three_d_optix_ao_num_rays": ("optix", "ao_num_rays"),
    "three_d_optix_ao_radius": ("optix", "ao_radius"),
    "three_d_optix_ambient_color": ("optix", "ambient_color"),
    "three_d_optix_rz_depth_of_field": ("optix", "rz_depth_of_field"),
    "three_d_optix_albedo_saturation": ("optix", "albedo_saturation"),
    "three_d_optix_albedo_brightness": ("optix", "albedo_brightness"),
    "three_d_optix_sphere_size_jitter": ("optix", "sphere_size_jitter"),
    "three_d_optix_use_curves": ("optix", "use_curves"),
    "three_d_optix_curve_length": ("optix", "curve_length"),
    "three_d_optix_curve_r0": ("optix", "curve_r0"),
    "three_d_optix_curve_r1": ("optix", "curve_r1"),
    "three_d_optix_sdf_enabled": ("optix", "sdf_enabled"),
    # (resolution_scale, rt_mode, rt_realtime_samples, rt_preview_spp,
    #  pt_firefly_clamp[_max] moved to the shared RenderingPrefs above.)
    # OptixPrefs (RT/PT)
    "three_d_rz_samples": ("optix", "rz_samples"),
    "three_d_pt_max_bounces": ("optix", "pt_max_bounces"),
    "three_d_pt_rr_start_depth": ("optix", "pt_rr_start_depth"),
    "three_d_pt_global_material": ("optix", "pt_global_material"),
    "three_d_pt_glossy_ior": ("optix", "pt_glossy_ior"),
    "three_d_pt_emission_intensity": ("optix", "pt_emission_intensity"),
    "three_d_pt_denoise_enabled": ("optix", "pt_denoise_enabled"),
    "three_d_rz_denoise_enabled": ("optix", "rz_denoise_enabled"),
    "three_d_pt_env_sky_nee": ("optix", "pt_env_sky_nee"),
    "three_d_pt_sun_exp": ("optix", "pt_sun_exp"),
    # Camera3DPrefs
    "three_d_fov": ("camera3d", "fov"),
    "three_d_aperture": ("camera3d", "aperture"),
    "three_d_focal_plane_depth": ("camera3d", "focal_plane_depth"),
    "three_d_move_speed": ("camera3d", "move_speed"),
    "three_d_rotate_speed": ("camera3d", "rotate_speed"),
    "three_d_orbit_center": ("camera3d", "orbit_center"),
    "three_d_orbit_rate": ("camera3d", "orbit_rate"),
    # UIWindowsPrefs
    "show_preferences_window": ("ui_windows", "show_preferences_window"),
    "show_controls_window": ("ui_windows", "show_controls_window"),
    "show_parameter_sweeps_window": ("ui_windows", "show_parameter_sweeps_window"),
    "show_tutorial_window": ("ui_windows", "show_tutorial_window"),
    "show_performance_window": ("ui_windows", "show_performance_window"),
    "show_generics_window": ("ui_windows", "show_generics_window"),
    "show_radio_window": ("ui_windows", "show_radio_window"),
    "show_plotting_window": ("ui_windows", "show_plotting_window"),
    "show_video_recording_window": ("ui_windows", "show_video_recording_window"),
    "show_scheduled_renders_window": ("ui_windows", "show_scheduled_renders_window"),
    "show_render_settings_window": ("ui_windows", "show_render_settings_window"),
    "show_config_clipboard_window": ("ui_windows", "show_config_clipboard_window"),
    "physics_tooltips_enabled": ("ui_windows", "physics_tooltips_enabled"),
    "mouse_mode": ("ui_windows", "mouse_mode"),
    "menu_close_threshold": ("ui_windows", "menu_close_threshold"),
    "physics_group_basics": ("ui_windows", "physics_group_basics"),
    "physics_group_forces": ("ui_windows", "physics_group_forces"),
    "physics_group_advanced": ("ui_windows", "physics_group_advanced"),
    "physics_group_additional": ("ui_windows", "physics_group_additional"),
    "physics_group_notes": ("ui_windows", "physics_group_notes"),
    "load_menu_core_open": ("ui_windows", "load_menu_core_open"),
    "load_menu_custom_open": ("ui_windows", "load_menu_custom_open"),
    "load_menu_advanced_open": ("ui_windows", "load_menu_advanced_open"),
}


def to_flat_dict(prefs: PreferencesState) -> dict:
    """Flatten a PreferencesState into the legacy flat-key dict.

    Used by services/render_spec.py to snapshot preferences as a flat dict.
    """
    out = {}
    for flat_key, (slice_attr, nested) in _FLAT_KEY_MAP.items():
        out[flat_key] = getattr(getattr(prefs, slice_attr), nested)
    return out


def set_flat(prefs: PreferencesState, key: str, value) -> None:
    """Set a single flat-keyed preference on the correct slice.

    No-op if the key is not a known flat preference key.
    """
    mapping = _FLAT_KEY_MAP.get(key)
    if mapping is None:
        return
    slice_attr, nested = mapping
    setattr(getattr(prefs, slice_attr), nested, value)


def from_flat_dict(data: dict) -> PreferencesState:
    """Build a PreferencesState from a legacy flat-key dict.

    Unknown keys are dropped (backward-compat filtering); missing keys keep
    their slice defaults.
    """
    prefs = PreferencesState()
    for key, value in data.items():
        set_flat(prefs, key, value)
    return prefs


def _from_nested_dict(data: dict) -> PreferencesState:
    """Build a PreferencesState from nested (current-format) JSON."""
    prefs = PreferencesState()
    for slice_attr in _SLICE_ATTRS:
        slice_data = data.get(slice_attr)
        if not isinstance(slice_data, dict):
            continue
        slice_obj = getattr(prefs, slice_attr)
        valid = set(type(slice_obj).__dataclass_fields__.keys())
        for k, v in slice_data.items():
            if k in valid:
                setattr(slice_obj, k, v)
    return prefs


def save_preferences(prefs: PreferencesState, filepath: Path | str = None) -> None:
    """Save preferences to a JSON file (nested per-slice format)."""
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(prefs)
    filepath.write_text(json.dumps(data, indent=2))


def preferences_from_dict(data: dict) -> PreferencesState:
    """Build a PreferencesState from a loaded JSON dict, running all migrations.

    Accepts both the current nested per-slice format and the legacy flat-key
    format. Shared by load_preferences() and EditorSaver so editor saves migrate
    exactly like the on-disk preferences.config.
    """
    # Nested format has the slice attrs as top-level keys; legacy format is flat.
    if isinstance(data, dict) and any(k in data for k in _SLICE_ATTRS):
        prefs = _from_nested_dict(data)
        _migrate_lighting_nested(prefs, data)
        _migrate_render_controls_nested(prefs, data)
    else:
        prefs = from_flat_dict(data)
    _migrate_renderer_selection(prefs, data)
    return prefs


def copy_preferences_into(target: PreferencesState, source: PreferencesState) -> None:
    """Copy every slice field from source onto target in place.

    Preserves target's object identity (and its slice objects' identities) so
    references held elsewhere (ui.state.preferences, etc.) see the update. Used
    when loading an EditorSave over the live preferences.
    """
    for slice_attr in _SLICE_ATTRS:
        src_slice = getattr(source, slice_attr)
        dst_slice = getattr(target, slice_attr)
        for fld in type(dst_slice).__dataclass_fields__:
            setattr(dst_slice, fld, getattr(src_slice, fld))


def load_preferences(filepath: Path | str = None) -> PreferencesState:
    """Load preferences from a JSON file. Returns default preferences if file doesn't exist.

    Accepts both the current nested per-slice format and the legacy flat-key
    format written before the preferences were split into slices.
    """
    if filepath is None:
        filepath = get_user_preferences_path()
    filepath = Path(filepath)
    if not filepath.exists():
        return PreferencesState()

    try:
        data = json.loads(filepath.read_text())
        return preferences_from_dict(data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Failed to load preferences from {filepath}: {e}")
        print("Using default preferences")
        return PreferencesState()


def _migrate_renderer_selection(prefs: PreferencesState, data: dict) -> None:
    """Migrate the pre-cleanup OptiX toggle to the new renderer enum.

    Before the unified ``rendering.renderer`` dropdown, OptiX was selected by
    ``optix.enabled`` (nested) / ``three_d_optix_enabled`` (flat). If a loaded
    prefs file predates ``renderer`` (so it stayed at the default 0=OpenGL) but
    had OptiX enabled, switch the renderer to Optix so the choice isn't lost.
    """
    has_renderer_key = (
        isinstance(data.get('rendering'), dict) and 'renderer' in data['rendering']
    ) or ('renderer' in data)
    if has_renderer_key:
        return  # File already uses the new enum; respect it verbatim.
    if prefs.optix.enabled:
        prefs.rendering.renderer = 1


def _migrate_lighting_nested(prefs: PreferencesState, data: dict) -> None:
    """Pull legacy per-renderer sun/sky into the shared LightingPrefs slice.

    Only for nested-format files predating LightingPrefs. If the file already
    has a ``lighting`` block, respect it. Otherwise adopt the OptiX values as
    canonical (per the cleanup decision), falling back to the legacy tracer
    fields for ``sky_intensity`` (OptiX had no such control).
    """
    if isinstance(data.get('lighting'), dict):
        return  # File already uses the shared slice.
    optix = data.get('optix', {}) if isinstance(data.get('optix'), dict) else {}
    tracer = data.get('tracer', {}) if isinstance(data.get('tracer'), dict) else {}
    lit = prefs.lighting
    if 'light_direction' in optix:
        lit.light_direction = list(optix['light_direction'])
    if 'light_color' in optix:
        lit.light_color = list(optix['light_color'])
    if 'light_intensity' in optix:
        lit.light_intensity = optix['light_intensity']
    if 'pt_sun_sampling' in optix:
        lit.nee = optix['pt_sun_sampling']
    if 'pt_photosphere' in optix:
        lit.photosphere = optix['pt_photosphere']
    if 'sky_color_top' in optix:
        lit.sky_color_top = list(optix['sky_color_top'])
    if 'sky_color_bottom' in optix:
        lit.sky_color_bottom = list(optix['sky_color_bottom'])
    if 'sky_intensity' in tracer:
        lit.sky_intensity = tracer['sky_intensity']


def _migrate_render_controls_nested(prefs: PreferencesState, data: dict) -> None:
    """Pull legacy per-renderer rt-mode / capture-spp / resolution / firefly
    into the shared RenderingPrefs slice (nested-format files predating them).

    The OptiX values are canonical (per the cleanup decision). If the file's
    ``rendering`` block already carries the new shared keys, respect it.
    """
    rendering = data.get('rendering', {}) if isinstance(data.get('rendering'), dict) else {}
    if 'rt_mode' in rendering:
        return  # File already uses the shared slice.
    optix = data.get('optix', {}) if isinstance(data.get('optix'), dict) else {}
    r = prefs.rendering
    if 'rt_mode' in optix:
        r.rt_mode = optix['rt_mode']
    if 'rt_realtime_samples' in optix:
        r.rt_samples = optix['rt_realtime_samples']
    if 'rt_preview_spp' in optix:
        r.capture_spp = optix['rt_preview_spp']
    if 'resolution_scale' in optix:
        r.render_resolution_scale = optix['resolution_scale']
    if 'pt_firefly_clamp' in optix:
        r.firefly_clamp = optix['pt_firefly_clamp']
    if 'pt_firefly_clamp_max' in optix:
        r.firefly_clamp_max = optix['pt_firefly_clamp_max']
