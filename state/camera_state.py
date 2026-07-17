from dataclasses import dataclass, field
import numpy as np


@dataclass
class CameraState:
    """State for camera position and rendering mode."""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    zoom: float = 1.0
    BRIGHTNESS: float = 1.  # Kept for backward compat, sourced from SimState

    # 3D camera (driven by ControllerCam + joystick)
    orbit_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # World-space point to orbit around
    orbit_rate: float = 0.0        # Auto-orbit speed (rad/physics-frame)
    orbit_angle: float = 0.0       # Accumulated orbit yaw angle (radians)
    orbit_pitch: float = 0.0       # Orbit elevation angle (radians)
    fov: float = 50.0              # Field of view (degrees)
    aperture: float = 0.0          # DOF lens radius (0 = pinhole, no DOF)
    focal_plane_depth: float = 5.0 # DOF focal plane distance
    move_speed: float = 2.0        # Joystick movement speed
    rotate_speed: float = 2.0      # Joystick rotation speed
    stereogram: bool = False       # Side-by-side stereo rendering
    eye_offset: float = 0.1        # Inter-eye separation (world units)
    stereo_toe_in: bool = False    # False = parallel eyes, True = converge (uses focal_plane_depth)
    stereo_wall_eye: bool = False  # False = cross-eye, True = wall-eye (screen halves swapped)
    optix_enabled: bool = False    # Transient per-frame flag: OptiX active (set from prefs.rendering.renderer; host may clear on failure)
    pathtracer_gas_time_ms: float = 0.0    # Path tracer GAS timing (for UI display)
    pathtracer_render_time_ms: float = 0.0 # Path tracer render timing (for UI display)
    pathtracer_sample_count: int = 0       # Current accumulation sample count (for UI display)
