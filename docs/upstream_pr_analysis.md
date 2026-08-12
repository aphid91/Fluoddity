# Upstream improvements: what's here and why

This document accompanies a PR of **generic improvements to existing Fluoddity** — performance and correctness work with no new features attached. It exists because a PR of this size deserves an explanation of *why* each change is worth adopting, and, just as importantly, an account of what was deliberately **left out**.

These changes were developed on a downstream branch alongside a good deal of experimental research. That research is not here. Everything below is an upgrade to machinery that already exists, so any branch should be able to merge it without inheriting a research agenda.

The two headline items are the 4-channel → 2-channel canvas texture (a large performance win for free) and a recording framerate that was quietly stuck at 50fps.

---

## 1. Fixes

### Recording fps 50 → 60

`FFmpegVideoRecorder` and `VidSaver` both defaulted to `fps=50`. This was an old attempt to reduce file sizes that was never revisited. Every video Fluoddity has exported since is subtly stuttery on 60Hz displays.

`utilities/ffmpeg_recorder.py`, `utilities/vid_saver.py`

### Brush kernel falloff

This one is a consequence of the canvas rewrite (§2) and is load-bearing for visual fidelity.

The old pipeline blended the brush with `SRC_ALPHA, ONE` where `alpha` was itself `kernel_func`. The kernel was therefore applied **twice** — once in the output value, once again by the blend — squaring it. The new pipeline blends with `ONE, ONE`, so that squaring has to become explicit:

```glsl
float p = clamp(trail_persistence, 0.001, 0.999);
brush_out = vel * (1.0 - p) * kernel_func * kernel_func;
```

Miss this and particle splats have visibly wrong falloff — softer and broader than before. The prescale is `(1-p)`, matching the old `canvas = blur(canvas)*p + (1-p)*brush` mix.

`shaders/brush.frag`

### Entity picker on non-square canvases

Click-to-pick did not correct for canvas aspect ratio, so on any non-square canvas it selected the wrong particle, with error growing toward the edges. Positions are now scaled by `sqrt(aspect)` / `sqrt(1/aspect)` before the distance search.

The same function also needed changing for the entity struct slim-down (§2), which is why both land together.

`services/entity_picker.py`, `command_handler.py`

---

## 2. Optimizations

### Canvas RGBA32F → RG32F

The canvas only ever stored two meaningful channels — X and Y velocity. The other two were written, blended, and sampled every frame while carrying nothing.

Halving the canvas format halves its VRAM, but the real win is bandwidth in the hottest read path in the engine: `get_can()` runs **twice per particle per physics step** (left and right sensor taps), at up to hundreds of thousands of particles and dozens of steps per frame.

Everything downstream follows mechanically: `getCan`/`getBlur` return `vec2`, the canvas fragment output is `vec2`, and the debug view reads `.rg`.

`sim.py`, `shaders/canvas.frag`, `shaders/brush.frag`, `shaders/entity_update.glsl`, `shaders/frame_assembly.frag`, `utilities/frame_assembler.py`, `camera.py`

### Brush middleman removed

The brush used to render into its own full-resolution RGBA32F texture, which `canvas.frag` then read back and mixed in:

```
brush pass → brush_tex → canvas.frag reads brush_tex → canvas
```

The intermediate texture serves no purpose. The brush now renders **directly into the canvas FBO** with additive blending, pre-scaled so the result is identical:

```
canvas.frag (decay only) → canvas ← brush pass (additive)
```

This deletes a full-resolution float texture, a per-step clear of it, and one texture read per canvas pixel per step. `canvas.frag`'s mix collapses to `can_color * trail_persistence`.

`sim.py`, `shaders/canvas.frag`, `shaders/brush.frag`

### Entity struct 48 → 32 bytes

```glsl
// before (48 bytes)          // after (32 bytes)
struct Entity {               struct Entity {
    vec2 pos;                     vec2 pos;
    vec2 vel;                     vec2 vel;
    float size;                   float hue;
    float cohort;                 float size;
    float padding[2];             float padding[2];
    vec4 color;               };
};
```

Two fields were redundant. `cohort` is a pure function of the entity index (`cohorts * index / ACTIVE_COUNT`), so it can be recomputed wherever needed instead of stored. `color` was a `vec4` of which three components were **hardcoded constants** — saturation `0.8`, brightness `1.0`, alpha `0.045` — set identically for every particle every frame. Only hue varied, so only hue is stored; the vertex shaders rebuild the `vec4`.

That is a 33% cut to entity-buffer traffic, which is read and written in full every physics step.

`sim.py`, `shaders/entity_update.glsl`, `shaders/brush.vert`, `shaders/cam_brush.vert`, `services/entity_picker.py`

### Per-frame uniform caching

Frame-constant uniforms were being re-uploaded on **every physics step**. With `speedmult` at 60–100 steps per rendered frame, dozens of `tryset()` calls — the physics-setting structs, appearance settings, boundary modes — ran 60–100× more often than they could possibly matter, since none of their source values change between steps of the same frame.

`apply_state()` now marks the uniforms dirty once per frame; the frame's first physics step uploads them and clears the flag. This is CPU-side, so it matters most exactly where it hurts most: high step counts.

**Invariant:** `apply_state()` must be called before stepping within a frame. The orchestrator already guarantees this, but it is now load-bearing rather than incidental, and is noted in the docstring.

Four uniforms are deliberately **excluded** from the cache because they genuinely vary per step: `frame_count`, `WRITE_RULES`, `fill_mode` (the caller gates it to step 0), and the multi-load weighted trail values (multi-load progress advances every step).

`sim.py`

---

## 3. Removals

Two things are deleted. Both are judgment calls rather than pure wins, so they are called out plainly.

### Emboss

Fully excised: the UI combo and two sliders (in both menus), three `SimState` fields, three `PhysicsConfig` fields, the `emboss()` / `gradient()` / `tiled_sample_uv_emboss()` functions in `frame_assembly.frag`, the `emboss_tex` plumbing through camera/assembler/runner, and two entries in `LOCKABLE_SIM_PARAMS`.

Emboss estimated a gradient from a texture's `.z` channel — a channel the 2-channel canvas no longer has. Keeping it would have meant reworking the effect around the new format. It was cut instead.

**Existing configs still load.** Legacy `.frs` files are parsed as before and the emboss fields are read and discarded rather than causing an error, on both the v6+ and older binary paths.

### `DEBUG - Brush` view

This one is forced: the view displayed `brush_tex`, and that texture no longer exists.

Its removal renumbers every view index above it — `0=can, 1=brush, 2=cam_brush, 3=tiled, 4=force, 5=strafe` becomes `0=can, 1=cam_brush, 2=tiled, 3=force, 4=strafe`. Hardcoded indices in `main.py`, `ui/core.py`, `ui/physics_window.py`, `ui/preferences_window.py`, `state/sim_state.py`, `camera.py`, and `frame_assembly.frag` all shift accordingly. Reviewers should look hardest here; it is the most mechanically error-prone part of the PR.

---

## 4. Minor additions

Three small things, none of which change behavior:

- **`tryset_mat3()`** in `utilities/gl_helpers.py` — 3×3 matrix uniform upload, mirroring `tryset()`'s graceful handling of optimized-away uniforms. No call sites yet.
- **`defines_prefix`** hook on the override-shader path, enabling `#define`-based shader variants via the existing `shader_prepend`.
- **FPS / frametime readout** in the Preferences window. Six lines, no state. Included so the performance claims above can be checked rather than taken on faith.

---

## 5. What was deliberately left out

The branch this came from contains substantially more. None of it is here, by design.

**New features.** A "Generics" window (8 live-coding scratch uniforms), a GPU histogram plotting system (`report()` in the physics shader feeding an SSBO and a plot window), and a "Cohort Confinement" physics parameter. Each is self-contained and arguably useful, but each is *new functionality* rather than an upgrade to existing functionality, and each carries UI surface, state fields, and config-format implications that a fork should get to opt into deliberately.

**A new feature system.** A lottery-based natural-selection system — per-pixel entity competition via atomic ticket writes, with winners mutating their rules and losers adopting the winner's, driving rule evolution over time. It spans three new compute shaders, a persistent-rules mode branching through the physics shader, a uint32 ticket texture, a new view mode, and reseed lifecycle logic.

It is also, candidly, **mid-experiment**: the payout shader currently contains a hardcoded early-return that excludes 70% of particles, a fully commented-out winner-mutation branch, and magic constants with alternatives commented beside them. It is not ready to be anyone else's dependency.

Worth noting for anyone who later wants it: the lottery's ticket write depends on the `entity_update → canvas` pass ordering established by the canvas rewrite here. This PR does not need the lottery, but a future lottery would sit on top of this PR.

**Exploratory aspect-ratio work.** A separate line of in-progress fixes to reticle and field-overlay positioning in `frame_assembly.frag`, still full of commented-out alternatives. The one piece that was solid — the entity picker correction — is included above; the rest is not.

---

## 6. Verifying this PR

Fluoddity has no automated tests; verification is manual.

**Performance.** Compare the FPS readout before and after at matched settings and high `speedmult`. The gain should be obvious rather than marginal.

**Visual fidelity** — the highest-risk area. Load the same `.frs` config before and after and compare side by side. Trails and splat falloff should look *the same*, not merely plausible; this is what the brush kernel fix exists to preserve. Sweep `TRAIL_PERSISTENCE` across its range, since the `(1-p)` prescale interacts with decay.

**View modes.** Cycle every entry. Confirm there is no Brush entry, each label matches what it shows, watercolor is enabled in Camera/Tiled and disabled elsewhere, tiling mode still engages and repositions the camera on exit, and both reticles land correctly in each mode.

**Entity picking.** With cohort sweeps active, pick a particle and confirm the reported cohort matches the particle actually clicked — this is where a wrong stride or a lost aspect correction shows up. Check on a non-square canvas specifically. Confirm cohort coloring works with `color_by_cohort` both on and off.

**Config compatibility.** Load a config saved before this change, including a legacy binary one, and confirm it loads cleanly with emboss fields discarded. Save and reload a new config.

**Uniform caching.** Drag physics sliders during playback; changes must take effect immediately. A stale dirty flag shows up as sliders that only apply after some unrelated interaction. Test with multi-load active and with parameter sweeps enabled, as both have per-step paths that bypass the cache.

**Regressions.** Motion blur on/off, strong determinism on/off, canvas draw/erase/fill, video recording (confirm 60fps output), screenshot.
