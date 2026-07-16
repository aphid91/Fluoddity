"""Xbox controller input: FPS-style ControllerCam and joystick polling.

Ported from Tracer/ui.py — same button/axis mappings, deadzone, and update logic.
"""
import glfw
import math
import numpy as np


class ControllerCam:
    """FPS-style camera with yaw/pitch control"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset camera to default position and orientation.

        Starts at [0, 0, -3] looking along +Z toward the simulation origin.
        """
        self.pos = np.array([0.0, 0.0, -3.0])
        self.yaw = 0.0      # Rotation around Y axis (radians)
        self.pitch = 0.0    # Rotation around X axis (radians), clamped to ±π/2
        self.fov = 50.
        self._update_vectors()

    def _update_vectors(self):
        """Update direction vectors from yaw and pitch angles."""
        # Direction vector (where camera is looking)
        self.dir = np.array([
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
            math.cos(self.pitch) * math.cos(self.yaw)
        ])

        # Right vector (perpendicular to dir in XZ plane)
        self.right = np.array([
            math.cos(self.yaw),
            0.0,
            -math.sin(self.yaw)
        ])

        # Up vector (cross product of right and dir)
        self.up = np.cross(self.right, self.dir)
        self.up = self.up / np.linalg.norm(self.up)

    def rotate(self, delta_yaw, delta_pitch):
        """Rotate camera by given angles, clamping pitch to ±π/2."""
        self.yaw += delta_yaw
        self.pitch = np.clip(self.pitch + delta_pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01)
        self._update_vectors()

    def move_xz(self, forward_amount, right_amount):
        """Move in the XZ plane relative to camera direction."""
        # Get forward direction projected onto XZ plane
        forward_xz = np.array([self.dir[0], 0.0, self.dir[2]])
        forward_len = np.linalg.norm(forward_xz)
        if forward_len > 0.001:
            forward_xz = forward_xz / forward_len
        else:
            forward_xz = np.array([0.0, 0.0, 1.0])

        # Move in XZ plane
        self.pos += forward_xz * forward_amount
        self.pos += self.right * right_amount

    def move_y(self, amount):
        """Move along the Y axis."""
        self.pos[1] += amount

    def get_position(self):
        """Get camera position."""
        return self.pos

    def get_view_vectors(self):
        """Get camera direction, right, and up vectors."""
        return self.dir, self.right, self.up


# Controller constants (Xbox-style)
MOVE_SPEED = 2.0
FAST_MULTIPLIER = 3.0
ROTATE_SPEED = 2.0
DEADZONE = 0.15

# Axis indices
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_LT = 4
AXIS_RT = 5

# Button indices
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
BUTTON_SELECT = 6
BUTTON_START = 7

# D-pad is reported as a hat, not a button. Values are a bitmask of directions.
HAT_DPAD = 0


def find_joystick():
    """Find the first connected joystick."""
    for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
        if glfw.joystick_present(jid):
            name = glfw.get_joystick_name(jid)
            if name:
                print(f"Found joystick {jid}: {name.decode() if isinstance(name, bytes) else name}")
            return jid
    return None


def apply_deadzone(value):
    """Apply deadzone to axis input."""
    if abs(value) < DEADZONE:
        return 0.0
    # Rescale to 0-1 range after deadzone
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


def process_controller_input(controller_cam, joystick_state, dt, *,
                              move_speed=MOVE_SPEED, rotate_speed=ROTATE_SPEED):
    """Update controller camera based on Xbox controller input.

    Args:
        controller_cam: ControllerCam instance to update.
        joystick_state: Mutable dict with 'joystick_id' and 'prev_buttons'.
        dt: Delta time in seconds.
        move_speed: Movement speed multiplier.
        rotate_speed: Rotation speed multiplier.
    """
    jid = joystick_state['joystick_id']

    # Check connection, try to reconnect if lost
    if jid is None or not glfw.joystick_present(jid):
        jid = find_joystick()
        joystick_state['joystick_id'] = jid
        if jid is None:
            return

    # Get joystick state
    axes_raw = glfw.get_joystick_axes(jid)
    buttons_raw = glfw.get_joystick_buttons(jid)
    hats_raw = glfw.get_joystick_hats(jid)

    if axes_raw is None or buttons_raw is None:
        return

    # GLFW returns (ctypes_pointer, count) tuple
    axes_ptr, axes_count = axes_raw
    buttons_ptr, buttons_count = buttons_raw

    if axes_count == 0 or buttons_count == 0:
        return

    # Extract values from ctypes pointers
    axes = [axes_ptr[i] for i in range(axes_count)]
    buttons = [buttons_ptr[i] for i in range(buttons_count)]

    hats = []
    if hats_raw is not None:
        hats_ptr, hats_count = hats_raw
        hats = [hats_ptr[i] for i in range(hats_count)]

    # Pad axes list if needed
    while len(axes) < 6:
        axes.append(0.0)

    # Pad buttons list if needed
    while len(buttons) < 16:
        buttons.append(0)

    # Button edge detection
    prev = joystick_state['prev_buttons']

    # Start button: reset camera
    if len(prev) > BUTTON_START:
        if buttons[BUTTON_START] and not prev[BUTTON_START]:
            print("Controller: Reset camera!")
            controller_cam.reset()

    # Select button edge detection (for tracer accumulation reset)
    joystick_state['select_pressed'] = bool(
        len(prev) > BUTTON_SELECT and buttons[BUTTON_SELECT] and not prev[BUTTON_SELECT]
    )

    # A button edge detection (cycle RT mode)
    joystick_state['cycle_rt_mode_pressed'] = bool(
        len(prev) > BUTTON_A and buttons[BUTTON_A] and not prev[BUTTON_A]
    )

    # B button edge detection (toggle pause)
    joystick_state['toggle_pause_pressed'] = bool(
        len(prev) > BUTTON_B and buttons[BUTTON_B] and not prev[BUTTON_B]
    )

    # X button edge detection (reset simulation)
    joystick_state['reset_pressed'] = bool(
        len(prev) > BUTTON_X and buttons[BUTTON_X] and not prev[BUTTON_X]
    )
    # Y button edge detection (randomize mutations)
    joystick_state['randomize_mutations_pressed'] = bool(
        len(prev) > BUTTON_Y and buttons[BUTTON_Y] and not prev[BUTTON_Y]
    )

    joystick_state['prev_buttons'] = buttons.copy()

    # D-pad edge detection (hat bitmask: HAT_UP/DOWN/LEFT/RIGHT can combine for diagonals)
    prev_hat = joystick_state.get('prev_hat', 0)
    hat = hats[HAT_DPAD] if len(hats) > HAT_DPAD else 0

    def hat_pressed(direction):
        return bool(hat & direction and not prev_hat & direction)

    # Up: select particle at screen center (equivalent to left click)
    joystick_state['dpad_pick_pressed'] = hat_pressed(glfw.HAT_UP)
    # Down: undo (equivalent to right click)
    joystick_state['dpad_undo_pressed'] = hat_pressed(glfw.HAT_DOWN)
    # Left: rack focus at screen center (equivalent to hovering center + pressing N)
    joystick_state['dpad_focus_pressed'] = hat_pressed(glfw.HAT_LEFT)

    joystick_state['prev_hat'] = hat

    # Right bumper held = fast mode
    speed_mult = FAST_MULTIPLIER if buttons[BUTTON_RB] else 1.0

    # Left stick - XZ movement
    left_x = apply_deadzone(axes[AXIS_LEFT_X])
    left_y = apply_deadzone(axes[AXIS_LEFT_Y])

    if left_x != 0 or left_y != 0:
        move_speed_dt = move_speed * speed_mult * dt
        # Y axis inverted (up = negative)
        controller_cam.move_xz(-left_y * move_speed_dt, left_x * move_speed_dt)

    # Right stick - rotation
    right_x = apply_deadzone(axes[AXIS_RIGHT_X])
    right_y = apply_deadzone(axes[AXIS_RIGHT_Y])

    if right_x != 0 or right_y != 0:
        rotate_speed_dt = rotate_speed * dt
        controller_cam.rotate(right_x * rotate_speed_dt, -right_y * rotate_speed_dt)

    # Triggers - Y movement
    lt = axes[AXIS_LT]
    rt = axes[AXIS_RT]

    # Normalize triggers: convert from [-1, 1] to [0, 1] if needed
    lt_normalized = (lt + 1.0) / 2.0 if lt < 0 else lt
    rt_normalized = (rt + 1.0) / 2.0 if rt < 0 else rt

    # Deadzone on the raw trigger input (not on the per-frame displacement):
    # gating y_movement after multiplying by move_speed*dt made low move speeds
    # fall below a fixed threshold and stutter as dt varied frame to frame.
    TRIGGER_DEADZONE = 0.05
    lt_normalized = lt_normalized if lt_normalized > TRIGGER_DEADZONE else 0.0
    rt_normalized = rt_normalized if rt_normalized > TRIGGER_DEADZONE else 0.0

    # Triggers are binary-feeling vs. the analog sticks, so run vertical movement
    # at half the stick move speed for finer low-speed control.
    TRIGGER_SPEED_SCALE = 0.5
    y_input = rt_normalized - lt_normalized
    if y_input != 0.0:
        y_movement = y_input * move_speed * speed_mult * dt * TRIGGER_SPEED_SCALE
        controller_cam.move_y(y_movement)

