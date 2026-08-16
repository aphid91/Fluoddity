# Streamline Audio — Handoff

Branch `streaming`. Last commit `e0a51b5`.

## What this is

Streamline particles are **read-only Fluoddity particles**: they read two
sensors and evaluate `calculate_entity_behavior` exactly as `entity_update`
does, but leave no trail, so they observe the simulation without perturbing
it. Their velocity drives an audio voice.

Key files:

| File | Role |
|---|---|
| `shaders/entity_physics.glsl` | Physics shared by `entity_update` and the tracer. Spliced via `shader_prepend`, not `#include`. |
| `shaders/streamline_trace.glsl` | The tracer. Steps particles, writes ring + audio lanes. |
| `shaders/streamline_audio_mix.glsl` | Sums per-voice lanes into the mono mix. |
| `services/streamline_service.py` | Dispatch scheduling, uniforms, draw. |
| `services/audio_service.py` | GPU ring, fences, limiter, rtmixer output. |
| `services/audio_capture.py` | Offline (recording) audio. |

`STREAMLINE_READONLY` makes `entity_physics.glsl` skip the entity buffer
declaration and expose `get_can_lerp()`.

## THE OPEN BUG — start here

**The audio does not interpolate across most canvas states at high physics
rates.** This is the "zippery above ~240 Hz" the user hears.

Per rendered frame, `sim.update()` runs *all* `speedmult` physics steps
back-to-back. The canvas ping-pongs between two textures throughout, so
`canvas` / `canvas_prev` only ever hold the **latest pair** (frames N and
N−1). Everything from N−speedmult to N−2 is overwritten before the tracer
looks at it.

Measured:

| speedmult | physics Hz | canvas frames advanced per render frame | frames never sensed |
|---|---|---|---|
| 1 | 60 | 1 | **0** |
| 10 | 600 | 10 | 9 |
| 25 | 1500 | 25 | **24** |

So at 600 Hz the audio smoothly interpolates one interval and **jumps over
nine**. The residual seam is therefore at the *render frame rate* (~60 Hz),
not the physics rate. speedmult=1 sounds clean because nothing is skipped.

**Proposed fix:** step the tracer *between* physics steps rather than after
all of them. `SimulationRunner` already has the right hook shape — see
`pre_record_hook`; an equivalent `post_physics_step_hook` was prototyped for
the video path and reverted. It belongs in the realtime path.

**Fallback already shipped:** Streamlines window → **Field Interpolation**
checkbox (`streamline.field_interpolation`) toggles the whole scheme off for
A/B.

## Second open item — video recording audio

Recorded audio sounds wrong: "chunky", not like the realtime preview. Root
cause is the same class of bug — all physics steps run, *then* the whole
video frame's ~800 samples are generated against one frozen canvas (a 60 Hz
staircase). An attempted fix was **fully reverted** (it broke realtime audio:
0 blk/s). Working tree is clean of it. Re-attempt only after the realtime
interleaving above is solved, since it is the same fix.

## Fixed this session (all committed, all verified)

1. **`b170311` Tracer had no rule.** `Sim.apply_rule()` only wrote
   `entity_update_program`, so the tracer's `target_rule` was all zeros — the
   sentinel meaning "generate a random rule". Streamers ran an unrelated rule
   the entire time. Twin-particle study on `Streamer_testing`: step-1 speed
   ratio **0.65 → 1.00**, direction cosine **−0.375 → +0.921**.
2. **`115cc60` 750 Hz tone.** The blend phase restarted at 0 every
   sub-dispatch, so the field ramped then snapped back — a sawtooth at the
   64-step sub-dispatch rate, fixed at 750 Hz across the whole 400–600 Hz
   working range. Phase now carries across sub-dispatches. **User confirmed
   eliminated.**
3. **`c4555da` Buzz while paused.** With the sim paused the two canvas
   textures freeze holding two *different* frames (measured max diff 0.00128),
   and the blend kept sweeping between them at the simulated physics rate.
   Now holds at the current frame once the canvas stops advancing.
4. **`c9b23b9` GLsync leak** (user-authored, verified): `pump()` gated its
   drain on ring space, so a full ring left fences undeleted forever. Also
   narrowed `ctx.memory_barrier()` (which is `GL_ALL_BARRIER_BITS`) to
   `GL_SHADER_STORAGE_BARRIER_BIT`.
5. **`2ccd6c8` Audio overproduction.** `blocks_wanted()` returned ring
   *space*; the readback frees slots in the same frame, so the tracer refilled
   space about to be consumed. Was producing 2.14 blocks/frame against 1.57
   consumed (**ratio 1.34, 213 overruns/3 s**); now paced by elapsed time at
   the sample rate (**ratio 0.999–1.003, zero overruns**).
6. **`ac15ca7`** Streamers upgraded to full read-only Fluoddity particles.
   Also **`642bdae`** extracted `entity_physics.glsl`; **`8a33a04`** multi-voice
   mix; **`a62e710`** settings persistence.

## Uncommitted / unproven

**Self-trail correction** is live in the working tree but **never
committed and never validated against a correct baseline.** Sliders `Self
Trail` / `Self Trail Spread` are visible in the UI.

Rationale: a real particle senses ink it deposited itself — measured, the
canvas at a particle's own position is **7.8× stronger** than at a random
point and **0.78 cosine-aligned** with its own velocity. A read-only streamer
sees a hole there. But everything measured about it was against a streamer
running the **wrong rule** (bug 1 above), so all those numbers are void.

There is also a stash: `stash@{0}` "self-trail correction WIP: accumulation
term + fencepost fix" — contains a fencepost fix (the deposit is made
*after* `entity_update` writes new pos/vel, so it should use post-step state)
and a steady-state accumulation term (×`1/(1-persistence)` ≈ 20).

**Recommendation:** re-measure whether any self-trail gap exists at all now
that the rule bug is fixed, before trusting or extending this. Consider
deleting the sliders if the gap is gone.

## Measurement warnings — read before trusting any A/B

The particle field is chaotic. Trajectory-level and spectral comparisons are
**noise-dominated** and produced actively misleading results repeatedly:

- Three runs of an identical config gave HF-energy deltas of **+21%, −136%,
  −8%**.
- With the canvas frozen, identical settings run twice differed by **0.306**
  in mean position, while every correction setting differed by ~0.29–0.30 —
  i.e. the effect was smaller than the noise floor.
- A per-speedmult "conclusive" result was later shown to be pure noise.

**What does work:** twin-particle studies (seed streamers at the exact state
of real particles, compare step-by-step), direct state inspection, and
arithmetic on the scheduling. When judging audio quality, hand it to the user
— their ear found both the 750 Hz tone and the physics-rate buzz that
spectral analysis could not resolve.

**Also:** compare streamers against the **right cohort**. Streamers are
hardcoded cohort 0; on a 64-cohort config, cohort speeds vary 10× and the
population average is meaningless. A "3.6× too fast" finding was entirely
this artifact.

## Test protocol the user prefers

Audio off, `steps_per_dispatch = 1`, `dispatch_hz = speedmult * 60` — one
tracer step per physics step removes every rate confound. Config
`Streamer_testing.json` (user configs dir) is ideal: 1 cohort, no mutation,
no hazard.

## Gotchas

- `sim.py` is user-owned — ask before restructuring. Additive changes are
  fine (`apply_physics_uniforms_to`, the `program=` arg on
  `_assign_physics_setting`).
- Pre-existing circular import: `services/` ↔ `ui/`. Import `ui` first in
  standalone scripts.
- Strong Determinism is forced on; the second canvas buffer supplies
  `canvas_prev`.
- `Initial Speed = 0` leaves streamers frozen forever — at rest,
  `safenorm(vel)` is zero so sensors collapse and no force is generated.
- Bash heredocs mangle `\n` inside Python string literals. Use the Edit tool
  for anything containing escapes.
