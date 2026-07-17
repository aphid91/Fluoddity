"""Shared cadence-lock scheduler for offline video capture.

Both offline video backends (volumetric tracer and OptiX path tracer) must
capture each output frame with the *same* temporal cadence so recordings look
identical regardless of renderer. This module owns that single source of truth.

The three UI sliders that drive capture:

- **Capture Physics Frequency** (``recording.motion_blur_samples``) — how many
  physics steps elapse per rendered output frame. Call this ``physics_rate``.
- **Capture SPP** (``rendering.capture_spp``) — how many samples are blended to
  form each output frame. This is *authoritative*: the plan always yields
  exactly this many samples, never more, never fewer.
- **Blur Quality** (``recording.recording_blur_quality``) — the physics-frame
  *stride*: only every ``stride``-th physics frame is eligible to be sampled.
  When recording motion blur is disabled, the stride is the full physics rate
  (one sharp slot per output frame).

The plan places the ``capture_spp`` samples onto cadence-locked *slots* — physics
frames at multiples of ``stride`` — distributing them as evenly as possible. It
always advances exactly ``physics_rate`` physics steps per output frame so
playback speed stays locked to Capture Physics Frequency, even when
``capture_spp`` is too small to touch every eligible slot (samples land on a
subset of slots; the remaining physics still runs, un-sampled).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureSlot:
    """One cadence-locked sampling point within an output frame.

    Attributes:
        physics_before: Physics steps to run *before* sampling this slot,
            counted from the start of the output frame. Slots are ordered, so
            the caller advances physics from the previous slot's count to this
            one (the delta is always a multiple of the stride, except that the
            very first slot may include an initial step).
        samples: How many samples (SPP) to accumulate at this slot.
    """
    physics_before: int
    samples: int


@dataclass(frozen=True)
class CapturePlan:
    """Full per-output-frame capture schedule from :func:`build_capture_plan`."""
    slots: tuple[CaptureSlot, ...]
    total_physics: int   # Always == physics_rate: physics to advance per frame.
    total_samples: int   # Always == capture_spp: samples per output frame.

    @property
    def stride(self) -> int:
        """Physics frames between eligible slots (informational)."""
        if len(self.slots) < 2:
            return self.total_physics
        return self.slots[1].physics_before - self.slots[0].physics_before


def build_capture_plan(physics_rate: int, capture_spp: int,
                        blur_quality: int, motion_blur: bool) -> CapturePlan:
    """Build the cadence-lock plan for one output frame.

    Args:
        physics_rate: Physics steps per output frame (Capture Physics Frequency).
        capture_spp: Total samples per output frame (Capture SPP) — authoritative.
        blur_quality: Physics-frame stride (Blur Quality). Ignored when
            ``motion_blur`` is False.
        motion_blur: Whether recording motion blur is enabled. When False the
            stride collapses to ``physics_rate`` (one sharp slot per frame).

    Returns:
        A :class:`CapturePlan` whose slots' samples sum to exactly
        ``capture_spp`` and whose ``total_physics`` equals ``physics_rate``.
    """
    physics_rate = max(1, int(physics_rate))
    capture_spp = max(1, int(capture_spp))

    if motion_blur:
        stride = max(1, int(blur_quality))
    else:
        stride = physics_rate

    # Number of cadence-locked slots that fit in the frame's physics span.
    eligible_slots = max(1, physics_rate // stride)
    # Never ask for more distinct slots than we have samples to place.
    used_slots = min(eligible_slots, capture_spp)

    slots: list[CaptureSlot] = []
    prev_samples_cumulative = 0
    for i in range(used_slots):
        # Even integer bucketing of capture_spp across used_slots (same scheme
        # the tracer used historically): slot i gets the difference of running
        # totals, so the buckets sum to exactly capture_spp.
        samples_cumulative = ((i + 1) * capture_spp) // used_slots
        samples = samples_cumulative - prev_samples_cumulative
        prev_samples_cumulative = samples_cumulative

        # Physics elapsed before this slot: land on stride multiples, phased so
        # that every slot sits on the SAME parity/phase of the physics clock.
        # Using (i + 1) * stride puts the first slot at `stride` (entities have
        # already moved) and the last at used_slots * stride <= physics_rate,
        # giving a clean even/odd cadence lock — essential for filtering the
        # every-other-frame oscillations this whole mechanism exists to remove.
        physics_before = min((i + 1) * stride, physics_rate)
        slots.append(CaptureSlot(physics_before=physics_before, samples=samples))

    return CapturePlan(
        slots=tuple(slots),
        total_physics=physics_rate,
        total_samples=capture_spp,
    )
