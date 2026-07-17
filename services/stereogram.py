"""
Stereogram: pure per-eye camera math for side-by-side stereo rendering.

Standalone and dependency-free (numpy only). Knows nothing about renderers,
ImGui, or app state — callers pass in the base camera vectors and receive an
:class:`EyeView` describing where to place the eye camera, which direction it
looks, its (half-width) aspect ratio, and which screen half to draw into.

All three rendering backends (glPoints, OptiX pathtracer, volrender) derive
their rays from the same inputs — camera ``pos``, ``dir``, ``up``, ``fov`` and
an ``aspect`` — so a single helper serves them all. Each backend renders twice
(once per eye) into its half of the screen.

Convention (matches the spec):
- ``side = -1`` is the LEFT eye; its camera is shifted to the RIGHT by
  ``eye_offset / 2`` and it fills the LEFT half of the screen.
- ``side = +1`` is the RIGHT eye; its camera is shifted to the LEFT and fills
  the RIGHT half.

Because each half occupies only half the horizontal resolution, the aspect is
``(full_width / 2) / height`` so a sphere stays a sphere (never squashed).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class StereoParams:
    """Stereogram settings, built once per frame from camera state."""
    enabled: bool = False
    eye_offset: float = 0.1      # world units, total inter-eye separation
    toe_in: bool = False         # False = parallel, True = converge on a plane
    convergence: float = 5.0     # distance to convergence plane (toe-in only)
    wall_eye: bool = False       # False = cross-eye, True = wall-eye (screen halves swapped)


@dataclass
class EyeView:
    """Resolved per-eye camera, ready to feed an existing render path."""
    pos: np.ndarray              # eye position (world)
    dir: np.ndarray             # unit look direction
    up: np.ndarray              # unit up vector (unchanged from base)
    fov: float                   # vertical fov, degrees (unchanged from base)
    aspect: float                # half-width aspect = (full_width/2) / height
    viewport: tuple              # (x, y, w, h) in full-framebuffer pixels


def eye_sides():
    """Return the eye sides in left-to-right order: (-1, +1)."""
    return (-1, +1)


def eye_camera(pos, dir_vec, up, fov, full_width, height, side, params):
    """Compute the per-eye camera for one half of a side-by-side stereogram.

    Args:
        pos: Base camera position (3,).
        dir_vec: Base unit look direction (3,).
        up: Base unit up vector (3,).
        fov: Vertical field of view, degrees.
        full_width: Full framebuffer width in pixels.
        height: Full framebuffer height in pixels.
        side: -1 for the LEFT eye (camera shifted right, left half of screen),
              +1 for the RIGHT eye (camera shifted left, right half).
        params: :class:`StereoParams`.

    Returns:
        :class:`EyeView`.
    """
    pos = np.asarray(pos, dtype=np.float32)
    d = np.asarray(dir_vec, dtype=np.float32)
    d = d / max(np.linalg.norm(d), 1e-8)
    u = np.asarray(up, dtype=np.float32)
    u = u / max(np.linalg.norm(u), 1e-8)

    right = np.cross(d, u)
    right = right / max(np.linalg.norm(right), 1e-8)

    # Left eye (side=-1) shifts the camera to the right (+right); right eye
    # (side=+1) shifts to the left (-right).
    eye_pos = pos - side * right * (params.eye_offset * 0.5)

    if params.toe_in:
        # Aim each eye at a shared convergence point on the base view axis.
        target = pos + d * params.convergence
        eye_dir = target - eye_pos
        eye_dir = eye_dir / max(np.linalg.norm(eye_dir), 1e-8)
    else:
        eye_dir = d

    half_w = max(1, full_width // 2)
    aspect = half_w / max(height, 1)

    # wall_eye swaps which screen half each eye draws into (parallax/eye-shift
    # direction above is unchanged) so cross-eyed viewers can switch to the
    # wall-eyed (diverging) viewing technique.
    screen_side = -side if params.wall_eye else side

    if screen_side < 0:
        viewport = (0, 0, half_w, height)                 # left half
    else:
        viewport = (full_width - half_w, 0, half_w, height)  # right half

    return EyeView(
        pos=eye_pos.astype(np.float32),
        dir=eye_dir.astype(np.float32),
        up=u.astype(np.float32),
        fov=fov,
        aspect=aspect,
        viewport=viewport,
    )


def params_from_camera_state(cam):
    """Build :class:`StereoParams` from a CameraState-like object.

    Small adapter so callers don't hand-copy fields. Reuses ``focal_plane_depth``
    as the toe-in convergence distance.
    """
    return StereoParams(
        enabled=getattr(cam, "stereogram", False),
        eye_offset=getattr(cam, "eye_offset", 0.1),
        toe_in=getattr(cam, "stereo_toe_in", False),
        convergence=getattr(cam, "focal_plane_depth", 5.0),
        wall_eye=getattr(cam, "stereo_wall_eye", False),
    )
