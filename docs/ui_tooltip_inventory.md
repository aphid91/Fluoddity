# UI Tooltip Inventory

Enumeration of every interactive ImGui control in the windows below: Physics Settings, Preferences, Render Settings, Config Clipboard, Radio, Screen Recording Controls, and the Main Menu Bar. For each control: whether it has a tooltip, the tooltip text, and the file/line where it's defined so it can be edited or a tooltip can be added.

Not covered (out of scope): Plotting, Generics, Scheduled Renders, Guide/Controls/Performance help windows, and any other window not listed above.

## How to read this doc

- **Location** points to the `imgui.*` widget call itself. For physics sliders, the tooltip text actually lives in `ui/physics_params.py` (the `description` field of the parameter definition), not inline at the widget call — see the Shared Helpers section.
- Conditional controls (only rendered under some UI state) are noted in the Control column.
- "Yes (delayed)" means the tooltip uses `_delayed_tooltip` (`ui/core.py:625`), which only fires after the mouse hovers **and stays still** (`HoveredFlags_.delay_normal | HoveredFlags_.stationary`). "Yes (immediate)" means it uses the plain `if imgui.is_item_hovered(): imgui.set_tooltip(...)` idiom, which fires as soon as the mouse enters the item — used throughout `ui/render_settings_window.py`.

---

## 0. Shared Helper / Wrapper Functions

These are called by multiple windows below. If a control's tooltip isn't inline at its widget call, it's almost certainly threaded through one of these.

| Helper | Location | Purpose | How tooltip text is supplied |
|---|---|---|---|
| `slider_float_with_range_menu(...)` | `ui/slider_widgets.py:24` | Wraps `imgui.slider_float` with jitter tint, right-click range/reset menu, alt-click lock. | Does not show the parameter's description itself — only a fixed jitter-explainer tooltip when jitter is active. The caller (`render_physics_slider`) separately calls `render_custom_tooltip(pdef.label, pdef.description)`. |
| `add_slider_context_menu(...)` | `ui/slider_widgets.py:99` | Right-click popup: Jitter slider, Min/Max fields, Reset Range, Reset Value buttons. | Jitter slider has its own `_delayed_tooltip` at `ui/slider_widgets.py:146`. Other popup items have none. |
| `render_sweep_buttons(param_name)` | `ui/slider_widgets.py:226` | X/Y/C sweep-toggle buttons next to a slider (shown only when Parameter Sweeps enabled). | None on any of the three buttons. |
| `render_range_adjust_buttons(...)` | `ui/slider_widgets.py:357` | Widen (`^`) / narrow (`v`) buttons that rescale a slider's range. | One shared `_delayed_tooltip` at `ui/slider_widgets.py:388` covering both buttons as a group. |
| `render_physics_slider(pdef)` | `ui/slider_widgets.py:390` | Top-level renderer used by `physics_window.py` for every Basics/Forces/Advanced slider; wires together lock, sweep buttons, range buttons, the slider, and the description tooltip. | Passes `pdef.label` / `pdef.description` (from `ui/physics_params.py`) to `render_custom_tooltip` at `ui/slider_widgets.py:447`. |
| `render_custom_tooltip(label, description)` | `ui/physics_tooltip.py:97` | Not a real ImGui tooltip — records the hovered slider/description for the physics tooltip companion window. | Text = whatever `description` the caller passes (for physics sliders, `pdef.description`). |
| `render_physics_tooltip()` | `ui/physics_tooltip.py:113` | Draws a custom borderless companion window (animated shader graphic + `imgui.text_wrapped(description)`) docked to the Physics Settings window's right edge. Only activates for the ~12 sliders with a shader mode in `PARAM_BY_LABEL` (`ui/physics_tooltip.py:79-89`), e.g. Global Force Mult, Strafe Power, Mutation Scale, Drag, Axial/Lateral Force. | — |
| `_delayed_tooltip(text)` | `ui/core.py:625` | Standard hover-and-hold tooltip used almost everywhere outside Render Settings. Only shows if hover is stationary. | Text passed directly by the call site. |
| Inline `if imgui.is_item_hovered(): imgui.set_tooltip(...)` | throughout `ui/render_settings_window.py` | Render Settings' own idiom — fires immediately on hover, no delay/stationary gate. | Literal string at the call site. |
| `lock_widget(pls, param_name, base_label, ...)` | `parameter_locks/widgets.py:41` | Context manager wrapping lockable widgets (red style + alt-click toggle). Not a tooltip mechanism itself. | N/A |
| `lock_begin_combo(pls, param_name, opened)` | `parameter_locks/widgets.py:73` | Alt-click handling for `begin_combo`-style widgets. | N/A |
| `_persisted_header(label, pref_field)` | `ui/render_settings_window.py:24` | Collapsing-header wrapper that persists open/closed state to preferences instead of imgui's `.ini`. | N/A |

**Doc note:** `ui/README.md` is stale — it references a `history_window.py`/`HistoryWindowMixin` that no longer exists; that code is now split into `ui/config_clipboard_window.py` and `ui/physics_tooltip.py`. It also doesn't mention `radio_window.py`, `render_settings_window.py`, `scheduled_renders_window.py`, or `generics_window.py`.

---

## 1. Physics Settings Window

`imgui.begin('Physics Settings', ...)` — `ui/physics_window.py:13`

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| "Appearance" (menu) | begin_menu | No | — | `ui/physics_window.py:221` |
| Color by Cohort | checkbox | Yes (delayed) | "Colors particles based on their cohort assignment\nrather than their behavior." | `ui/physics_window.py:231-235` |
| Hue Sensitivity (shown only if not Color by Cohort) | slider_float | Yes (delayed) | "Controls color variation based on particle velocity." | `ui/physics_window.py:239-243` |
| Mutation Seed: #XXXX (alt-clickable text, not a real widget) | text / lock target | No | — | `ui/physics_window.py:22-32` |
| "Basics" | collapsing_header | No | — | `ui/physics_window.py:37` |
| Sensor Gain | slider (custom tooltip) | Yes | "Determines how strongly particles respond to sensor input. Higher values make particles more reactive to the trails they sense on the Canvas." | `ui/physics_params.py:47-51` |
| Sensor Angle | slider (custom tooltip) | Yes | "Sets the angular offset of particle sensors from their forward direction. Determines whether particles are 'looking ahead' or 'looking behind'." | `ui/physics_params.py:52-57` |
| Sensor Distance | slider (custom tooltip) | Yes | "Determines distance between a particle's center and where it reads the trail information from Canvas. Longer distances tend to create larger scale patterns." | `ui/physics_params.py:58-62` |
| Mutation Scale | slider (custom tooltip, hide_jitter) | Yes | "Controls the size of the random mutations applied to a rule when a new particle is clicked. At 0, every particle will behave exactly like the selected particle." | `ui/physics_params.py:63-68` |
| "Forces" | collapsing_header | No | — | `ui/physics_window.py:46` |
| Global Force Mult | slider (custom tooltip) | Yes | "Scales axial and lateral forces applied to particles, and scales strafe power. Often tuned in the opposite direction to Sensor Gain and Drag to offset exploding/vanishing particle speed." | `ui/physics_params.py:71-75` |
| Drag | slider (custom tooltip, -1..1) | Yes | "Each physics update, particle velocity is multiplied by drag like so:   vel = vel*drag + forces; So drag less than 1 means particles are being slowed down. Powerful (<0.5) drag values can prevent energetic systems from 'blowing up'" | `ui/physics_params.py:76-81` |
| "Advanced" | collapsing_header | No | — | `ui/physics_window.py:55` |
| Axial Force | slider (custom tooltip) | Yes | "Controls the strength of forces applied parallel to the direction of travel: acceleration and braking" | `ui/physics_params.py:84-88` |
| Lateral Force | slider (custom tooltip) | Yes | "Controls the strength of forces applied perpendicular to the direction of travel: turning left and right." | `ui/physics_params.py:89-93` |
| Strafe Power | slider (custom tooltip) | Yes | "Controls particle movement without applying forces to velocity. 'Strafe' is a vector added directly to position each frame, like a little hop. Strafe power scales with Axial, Lateral, and Global force multipliers." | `ui/physics_params.py:94-98` |
| Trail Persistence | slider (custom tooltip, 0..1) | Yes | "Controls how long particle trails remain visible. Higher values create longer-lasting trails, lower values make trails fade quickly. Values close to 1.0 tend to create 'sharper' more stable patterns. " | `ui/physics_params.py:99-104` |
| Trail Diffusion | slider (custom tooltip, 0..1) | Yes | "Controls how quickly particle trails spread out and blend together." | `ui/physics_params.py:105-110` |
| Hazard Rate | slider (power-scaled, exponent 3.0, hide_jitter) | Yes | "Probability per frame that particles reset to initial conditions. Gives particles a probabalistic 'lifetime' after which they reset." | `ui/physics_params.py:111-118` |
| "Additional Settings" | collapsing_header | No | — | `ui/physics_window.py:64` |
| Boundary Conditions | begin_combo | Yes (per-option) | Bounce: "Particles bounce off the edges of the canvas" · Reset: "Particles are reset to their initial conditions when leaving the canvas" · Wrap: "Particles wrap seamlessly to the other side of the canvas" | `ui/physics_window.py:69-86` |
| Initial Conditions | begin_combo | Yes (per-option) | Flat Grid: "Particles start in a flat grid on the floor plane, organized by cohort" · Random: "Particles are spread uniformly across the canvas" · Ring: "Particles start distributed around a circle, organized by cohort" · 3d Grid: "Particles start in a 3D grid filling the volume, organized by cohort" | `ui/physics_window.py:91-109` |
| Initialization Spacing | slider_float | Yes | "Scales the initial spacing between cohorts.\nGrid modes: shrinks the gaps between globs (glob size unchanged);\nat 0 all globs spawn stacked at the center.\nRing / Random: multiplies the overall spawn size." | `ui/physics_window.py:112-117` |
| Number of Cohorts | slider_int (1-144) | Yes | "Each particle is assigned to a cohort. Each cohort shares behavior:\neach cohort has a distinct mutation." | `ui/physics_window.py:120-125` |
| Gravity Force | slider_float (-1..1) | Yes | "Gravity-like force applied to particles.\nLogarithmic: force grows ~10x per quarter of slider travel,\nwith a dead-zone at center." | `ui/physics_window.py:132-136` |
| Gravity Strafe | slider_float (-1..1) | Yes | "Gravity-like strafe (direct position offset) applied to particles.\nLogarithmic: strafe grows ~10x per quarter of slider travel,\nwith a dead-zone at center." | `ui/physics_window.py:139-143` |
| Disable Symmetry | checkbox | Yes | "Allow particles to display \"right / left handed\" behavior,\nleading to clockwise/counterclockwise bias.\nTurn it on to see why we go through trouble\nof calculating \"mirror world\" behavior in entity_update.glsl" | `ui/physics_window.py:148-152` |
| Absolute Orientation | combo (Off / Y axis / Radial) | Yes | "What direction are particles 'facing'? Which way is 'up'?\nOff: use particle velocity\nY axis: align to y axis\nRadial: align to center of canvas" | `ui/physics_window.py:155-160` |
| Orientation Mix (only if Absolute Orientation != Off) | slider_float (0..1) | Yes | "Blend factor for orientation calculations (0.0 = velocity only, 1.0 = full absolute orientation)" | `ui/physics_window.py:163-169` |
| Parameter Sweeps | checkbox | Yes | f"Enable parameter sweeps to vary physics across the canvas.\nPress {sweep_key} to toggle. See Help -> Parameter Sweeps for details." | `ui/physics_window.py:174-179` |
| "Notes" | collapsing_header | No | — | `ui/physics_window.py:183` |
| Notes text field | input_text_multiline | Yes | "Optional notes to save with this config.\nThese will be saved when you save the config.\nEnter to finish editing, Ctrl+Enter for newline." | `ui/physics_window.py:187-196` |

**Per-slider extras** (Basics/Forces/Advanced sliders only, via `render_physics_slider`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| X / Y / C sweep buttons (shown when Parameter Sweeps enabled) | button ×3 | No | — | `ui/slider_widgets.py:255,281,307` |
| Widen (`^`) / Narrow (`v`) range buttons | button ×2 | Yes (one tooltip for the pair) | "Up arrow widens slider range. Down arrow narrows range" | `ui/slider_widgets.py:378,382,388` |

**Right-click context menu** (every Basics/Forces/Advanced slider, via `add_slider_context_menu`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Jitter slider | slider_float (0-2) | Yes | "Adds random jitter to this setting per-particle per-frame.\nOften results in a softer, fuzzier look." | `ui/slider_widgets.py:142-146` |
| Min | input_float | No | — | `ui/slider_widgets.py:153` |
| Max | input_float | No | — | `ui/slider_widgets.py:154` |
| Reset Range to Default | button | No | — | `ui/slider_widgets.py:166` |
| Reset value to '\<file\>' / Reset value to defaults | button | No | — | `ui/slider_widgets.py:179` |

---

## 2. Preferences Window

`imgui.begin("Preferences", True)` — `ui/preferences_window.py:18`

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Particle Count | input_int (commits on Enter) | Yes | "Number of active particles. Takes effect on Enter.\nMore particles = more VRAM (32 bytes each)." | `ui/preferences_window.py:25-36` |
| Canvas Resolution | input_int | Yes | "Cubic canvas dimension (W=H=D) for 3D trail textures.\nTakes effect on Enter. 6 textures at dim^3 * 4 bytes each." | `ui/preferences_window.py:39-50` |
| Rate (disabled/locked while recording) | slider_int (1-30, or 1-6 while recording) | Yes | "EXPENSIVE- Multiple physics steps can be calculated each\nrender frame and blended together for faster physics.\nMotion blur can be costly for high frequencies,\ntry turning it off if things feel sluggish." | `ui/preferences_window.py:70-91` |
| Motion Blur (disabled during recording) | checkbox | Yes | Same text as Rate (above) | `ui/preferences_window.py:97-101` |
| Blur Quality (shown only if Motion Blur enabled) | slider_int (1-20) | Yes | "Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance" | `ui/preferences_window.py:116-122` |
| Physics Tooltips | checkbox | Yes | "Enable verbose tooltip and vector diagram for physics sliders." | `ui/preferences_window.py:140-144` |

Note: a "Mouse Mode" combo exists in source (`ui/preferences_window.py:130-136`) but is commented out / inactive — not a live control.

---

## 3. Render Settings Window

`imgui.begin("Render settings", True)` — `ui/render_settings_window.py:40`. Tooltips here consistently use the **immediate** `is_item_hovered()` + `set_tooltip()` idiom rather than `_delayed_tooltip`.

**Top-level (always shown):**

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Renderer (OpenGL / Optix) | combo | Yes | "OpenGL: GL points (Pathtrace Off) or the volumetric path tracer.\nOptix: the RTX path tracer (rasterize / path-trace modes).\nRequires an NVIDIA RTX GPU with OptiX/CUDA installed for Optix." | `ui/render_settings_window.py:69-76` |
| Brightness | slider_float (0.01-2.5) | Yes | "Global brightness multiplier for the output." | `ui/render_settings_window.py:81-84` |
| Tonemap Softness | slider_float (0.1-4.0) | Yes | "Controls highlight compression (asinh stretch).\nLow = more linear (brighter highlights).\nHigh = more logarithmic (reveals faint detail)." | `ui/render_settings_window.py:85-91` |

**Camera section** (header "Camera"):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| FOV | slider_float (10-95 deg) | No | — | `ui/render_settings_window.py:161-162` |
| Aperture | slider_float (0-0.15) | No | — | `ui/render_settings_window.py:163-164` |
| Focal Depth | slider_float (0.1-10.0) | No | — | `ui/render_settings_window.py:165-166` |
| Move Speed | slider_float (0.1-5.0) | No | — | `ui/render_settings_window.py:167-168` |
| Rotate Speed | slider_float (0.1-5.0) | No | — | `ui/render_settings_window.py:169-170` |
| Orbit Center | drag_float3 | No | — | `ui/render_settings_window.py:171-176` |
| Orbit Rate | slider_float (-0.02 to 0.02) | No | — | `ui/render_settings_window.py:177-178` |
| Stereogram | checkbox | No | — | `ui/render_settings_window.py:181` |
| Eye Offset (if Stereogram) | slider_float (0-0.5) | No | — | `ui/render_settings_window.py:183-184` |
| Parallel | radio_button | No | — | `ui/render_settings_window.py:186-187` |
| Toe-in | radio_button | Conditional — Yes, only when this option is selected | "Eyes converge on the Focal Depth plane" | `ui/render_settings_window.py:189-192` |

**Lighting section** (header "Lighting"):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Light Dir | drag_float3 (-1..1) | No | — | `ui/render_settings_window.py:202-205` |
| Light Color | color_edit3 | No | — | `ui/render_settings_window.py:206-207` |
| Intensity | slider_float (0-20) | No | — | `ui/render_settings_window.py:208-209` |
| Enable NEE | checkbox | Yes | "Next Event Estimation: trace a shadow ray toward the\nlight for direct lighting. In rasterize mode, turning\nthis off leaves only the ambient + AO term." | `ui/render_settings_window.py:211-216` |
| Cos-lobe Sky (OptiX only) | checkbox | Yes | "Replace legacy directional sun + gradient sky\nwith a cosine-lobe environment model.\nSky color controls hemisphere glow,\nsun direction/color/intensity control sun disk." | `ui/render_settings_window.py:218-226` |
| Sun Sharpness (if Cos-lobe Sky) | slider_float (1-256) | Yes | "Exponent of the sun's cosine-power lobe.\nHigher = tighter, sharper sun disk." | `ui/render_settings_window.py:227-234` |
| Photosphere | checkbox | Yes | "Use equirectangular environment map for the sky\n(queried on primary-ray miss, i.e. the background)." | `ui/render_settings_window.py:235-240` |

**Sky section** (header "Sky"):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Sky Top | color_edit3 | No | — | `ui/render_settings_window.py:255` |
| Sky Bottom | color_edit3 | No | — | `ui/render_settings_window.py:256` |
| Sky Intensity (OpenGL renderer only) | slider_float (0-5) | No | — | `ui/render_settings_window.py:257-260` |

**Bloom controls** (shared, appears in Post-Process headers of both renderers):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Bloom | checkbox | Yes | "Add a glow effect around bright areas." | `ui/render_settings_window.py:96-98` |
| Threshold (if Bloom enabled) | slider_float (0-2) | Yes | "Brightness cutoff for bloom extraction.\nLower = more glow everywhere." | `ui/render_settings_window.py:101-103` |
| Intensity (Bloom) | slider_float (0-3) | Yes | "Strength of the bloom glow." | `ui/render_settings_window.py:104-106` |
| Radius (Bloom) | slider_float (0.1-3) | Yes | "Spread of the bloom blur kernel." | `ui/render_settings_window.py:107-109` |

**Firefly clamp** (shared, Post-Process):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Firefly Clamp | checkbox | No | — | `ui/render_settings_window.py:151` |
| Firefly max value field (if enabled) | drag_float | No | — | `ui/render_settings_window.py:154-155` |

**Pathtrace mode row** (shared):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| "Pathtrace: Off / N spp / Accumulate" cycle button | button | No | — | `ui/render_settings_window.py:124-125` |
| Samples slider (OptiX, path-trace mode) | slider_int (1-8) | No | — | `ui/render_settings_window.py:127-129` |
| Samples slider (OptiX, rasterize mode) | slider_int (1-8) | Yes | "Rasterize samples per frame (averaged before denoise)" | `ui/render_settings_window.py:133-137` |

**Resolution Scale row** (shared):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Resolution Scale | input_float | No | — | `ui/render_settings_window.py:142-146` |

**OptiX-specific settings** (`renderer == 1`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Capture SPP | slider_int (1-128) | No | — | `ui/render_settings_window.py:272` |
| Re-render Preview | button | No | — | `ui/render_settings_window.py:276-277` |
| "Geometry" header | collapsing_header | No | — | `ui/render_settings_window.py:310` |
| Sphere Scale | slider_float (0.1-10.0x) | No | — | `ui/render_settings_window.py:311-313` |
| Sphere Jitter (if not using Curves) | slider_float (0-1) | Yes | "Per-sphere radius jitter to reduce banding artifacts" | `ui/render_settings_window.py:315-319` |
| Curves | checkbox | Yes | "Render entities as round linear curves oriented along velocity" | `ui/render_settings_window.py:321-323` |
| Curve Length (if Curves) | slider_float (0-10) | No | — | `ui/render_settings_window.py:325-326` |
| Curve R0 (if Curves) | slider_float (0.01-5.0) | No | — | `ui/render_settings_window.py:327-328` |
| Curve R1 (if Curves) | slider_float (0.01-5.0) | No | — | `ui/render_settings_window.py:329-330` |
| Enable Dish | checkbox | No | — | `ui/render_settings_window.py:332` |
| "Material" header | collapsing_header | No | — | `ui/render_settings_window.py:335` |
| BRDF (Lambert/Glossy/Mirror) | combo | No | — | `ui/render_settings_window.py:336-338` |
| Glossy IOR (if Glossy) | slider_float (1.0-3.0) | No | — | `ui/render_settings_window.py:339-341` |
| Albedo Saturation | slider_float (0-1) | No | — | `ui/render_settings_window.py:342-343` |
| Albedo Brightness | slider_float (0-1) | No | — | `ui/render_settings_window.py:344-345` |
| "Rasterize" header | collapsing_header | No | — | `ui/render_settings_window.py:352` |
| Ambient (rasterize) | slider_float (0-1) | No | — | `ui/render_settings_window.py:356-357` |
| Ambient Color | color_edit3 | Yes | "Ambient tint (scaled by Ambient), modulated by AO" | `ui/render_settings_window.py:358-361` |
| Depth of Field | checkbox | Yes | "Use the thin-lens camera (Aperture / Focal Depth in the\nCamera section) in rasterize mode. Nearly free with the denoiser on." | `ui/render_settings_window.py:363-368` |
| Ambient Occlusion (rasterize) | checkbox | No | — | `ui/render_settings_window.py:370-371` |
| AO Rays (if AO enabled) | slider_int (1-16) | No | — | `ui/render_settings_window.py:373-374` |
| AO Radius (if AO enabled) | slider_float (0.01-5.0) | No | — | `ui/render_settings_window.py:375-376` |
| "Path Trace" header | collapsing_header | No | — | `ui/render_settings_window.py:381` |
| Max Bounces | drag_int (0-64) | Yes | "0 = unbounded (Russian roulette only)" | `ui/render_settings_window.py:384-387` |
| RR Start Depth | slider_int (1-16) | No | — | `ui/render_settings_window.py:388-389` |
| Emission Intensity | slider_float (0-100) | Yes | "Radiance multiplier for emissive entities (negative hue)" | `ui/render_settings_window.py:390-394` |
| "Post-Process" header | collapsing_header | No | — | `ui/render_settings_window.py:399` |
| Denoise (path trace) | checkbox | No | — | `ui/render_settings_window.py:402-403` |
| Denoise (rasterize) | checkbox | No | — | `ui/render_settings_window.py:404-405` |
| (+ shared Firefly Clamp & Bloom, see above) | | | | `ui/render_settings_window.py:400,407` |

**OpenGL-specific settings** (`renderer == 0`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Capture SPP | slider_int (1-512) | No | — | `ui/render_settings_window.py:437` |
| Re-render Preview | button | No | — | `ui/render_settings_window.py:441-442` |
| "Path Trace" header | collapsing_header | No | — | `ui/render_settings_window.py:469` |
| Colored Extinction | checkbox | No | — | `ui/render_settings_window.py:470-471` |
| Albedo RGB (if Colored Extinction) | color_edit3 | No | — | `ui/render_settings_window.py:473` |
| Ext Saturation (if Colored Extinction) | slider_float (0-1) | No | — | `ui/render_settings_window.py:474-475` |
| Ext Brightness (if Colored Extinction) | slider_float (0-1) | No | — | `ui/render_settings_window.py:476-477` |
| Extinction RGB (else) | color_edit3 | No | — | `ui/render_settings_window.py:479` |
| Albedo Saturation (else) | slider_float (0-1) | No | — | `ui/render_settings_window.py:480-481` |
| Albedo Brightness (else) | slider_float (0-1) | No | — | `ui/render_settings_window.py:482-483` |
| Density Scale | slider_float (0.00001-10.0) | No | — | `ui/render_settings_window.py:486-487` |
| Scattering (g) | slider_float (-1..1) | Yes | "HG phase: -1 back, 0 isotropic, +1 forward" | `ui/render_settings_window.py:488-490` |
| Emission | drag_float (0-100) | Yes | "Self-emission intensity (0 = off)" | `ui/render_settings_window.py:491-494` |
| Max Bounces | drag_int (0-64) | Yes | "0 = unbounded (Russian roulette only)" | `ui/render_settings_window.py:495-497` |
| Enable Dish | checkbox | No | — | `ui/render_settings_window.py:499` |
| "Grid Resolutions" header | collapsing_header | No | — | `ui/render_settings_window.py:506` |
| Density (2^n) | slider_int (5-10) | Yes | dynamic, e.g. "512x512x512  (1024 MB)" | `ui/render_settings_window.py:507-511` |
| Color (2^n) | slider_int (5-10) | Yes | dynamic, e.g. "512x512x512  (2048 MB, 2 channels)" | `ui/render_settings_window.py:512-516` |
| Majorant (2^n) | slider_int (3-8) | Yes | dynamic, e.g. "64x64x64" | `ui/render_settings_window.py:517-521` |
| "Post Process" header | collapsing_header | No | — | `ui/render_settings_window.py:529` |
| (+ shared Firefly Clamp & Bloom) | | | | `ui/render_settings_window.py:530-531` |

---

## 4. Config Clipboard Window

`imgui.begin("Config Clipboard - EXPERIMENTAL", True)` — `ui/config_clipboard_window.py:41`

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Checkpoint entry (one per checkpoint, newest first) | selectable | No | — | `ui/config_clipboard_window.py:63-68` |
| Delete ("X") button per entry | small_button | No | — | `ui/config_clipboard_window.py:83-86` |
| Rename field (right-click popup on an entry) | input_text | No | — | `ui/config_clipboard_window.py:96-103` |

Static (non-interactive) text: "Press Ctrl+C to add a checkpoint" (`:47`), "No checkpoints yet" (`:51`).

---

## 5. Radio Window

`imgui.begin("Radio", True)` — `ui/radio_window.py:10`

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Radio enabled | checkbox | No | — | `ui/radio_window.py:16-19` |
| Target Frequency | slider_float (-15.0 to 15.0) | No | — | `ui/radio_window.py:23-27` |
| Bandwidth | slider_float (0.0 to 2.0) | No | — | `ui/radio_window.py:31-35` |

Despite the window's name, it has no `imgui.radio_button` widgets — it's a frequency-band visibility filter. (The Extras menu entry that opens it does have a tooltip — see Main Menu, item 26.) The only true radio-button pair among the surveyed windows is "Parallel"/"Toe-in" in Render Settings → Camera → Stereogram.

---

## 6. Screen Recording Controls Window

`imgui.begin("Screen Recording", True)` — `ui/help_windows.py:228`. Opened via Extras → "Screen Recording Controls" (Main Menu, item 22 below).

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Video End Frame | input_int | Yes | "Target frame for video to end on.\nWhen set, recording will be delayed until the\ncalculated start frame is reached.\nSet to 0 to start recording immediately." | `ui/help_windows.py:254-258` |
| Video Length | drag_float ("%.0f seconds") | Yes | "After Video reaches this length, the recording will be stopped" | `ui/help_windows.py:263-273` |
| Capture Physics Frequency (disabled while recording) | slider_int (1-100) | Yes | "Physics steps per frame for video/screenshots.\nHigher values = faster physics with smoother motion blur.\nAlso determines screenshot exposure (# of samples to blend together)." | `ui/help_windows.py:281-288` |
| Motion Blur (Recording) | checkbox | Yes | "Enable motion blur during video recording.\nThis setting overrides the Motion Blur checkbox in Preferences while recording." | `ui/help_windows.py:315-319` |
| Blur Quality (Recording) (if Motion Blur (Recording) enabled) | slider_int (1-20) | Yes | "Motion Blur can be expensive at high frequencies,\nskip some frames to improve performance.\nThis setting overrides the Blur Quality slider in Preferences while recording." | `ui/help_windows.py:331-337` |
| Downsample Resolution Factor | input_int | Yes | "Set to '2' to render a video at half resolution." | `ui/help_windows.py:345-346` |
| Filename | input_text (max 256) | Yes | "Defaults to 'animation' if left empty. Saves to documents/Fluoddity/ Video filenames get timestamps appended" | `ui/help_windows.py:349-354` |

Non-interactive display elements also present: recording status text, current frame counter, and a computed "Frame Range: X --- Y" text that itself has a `_delayed_tooltip` (`ui/help_windows.py:307-311`): "Estimated recording range based on current settings.\nTotal simulation frames: {n}\n({max_frames} output frames x {motion_blur_samples} physics steps)".

---

## 7. Main Menu Bar

`imgui.begin_main_menu_bar()` — `ui/menu_bar.py:13`

**File menu** (`ui/menu_bar.py:24`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| New | menu_item | Yes | "Start a fresh config. Loads from _Default" | `ui/menu_bar.py:33-37` |
| Save... | menu_item | Yes | "Save the current physics settings including particle rules." | `ui/menu_bar.py:41-45` |
| Load (submenu, live-preview list of config files) | begin_menu | No | — | `ui/menu_bar.py:48-94` |

**Editor menu** (`ui/menu_bar.py:98`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Preferences (toggle) | menu_item | No | — | `ui/menu_bar.py:107-108` |
| Render settings (toggle) | menu_item | Yes | "Per-renderer controls: RT mode, capture, camera,\nmedium/geometry, lighting, sky, and post-process.\nShows the active renderer's settings (Preferences -> Renderer)." | `ui/menu_bar.py:111-113` |
| Save Editor Settings... | menu_item | Yes | "Save all non-physics settings (preferences, render settings,\nwindow visibility, and window/docking layout)." | `ui/menu_bar.py:118-123` |
| Load Editor Settings (submenu, `.editor.json` list + per-entry delete button) | begin_menu | No | — | `ui/menu_bar.py:126-157` |

**Top-level items** (not in a submenu):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Show/Hide Windows (X) | menu_item | No | — | `ui/menu_bar.py:162-163` |

**Reset... menu** (`ui/menu_bar.py:166`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Revert to '{project}' (dynamic label) | menu_item | Yes | dynamic, e.g. "Equivalent to File -> Load {project}" | `ui/menu_bar.py:176-182` |
| Reset all slider ranges to defaults | menu_item | No | — | `ui/menu_bar.py:185-187` |
| Reset all parameter sweeps | menu_item | Yes | "Set all parameter sweeps to 'off'." | `ui/menu_bar.py:190-196` |
| Reset all UI settings | menu_item | Yes | "Restore all preferences and ui state to factory settings. \nEquivalent to deleting preferences.config, or running this\nprogram for the first time. Physics config saves are not affected." | `ui/menu_bar.py:199-204` |
| Reset camera | menu_item | Yes | "Return camera to default position and zoom level." | `ui/menu_bar.py:207-209` |
| Reset Canvas | menu_item | Yes | "Clear the trail canvas to zero." | `ui/menu_bar.py:212-214` |

**Locks menu** (only visible if parameter locks enabled, `ui/menu_bar.py:221`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Unlock Everything / Lock Everything (dynamic label) | menu_item | No | — | `ui/menu_bar.py:230-235` |
| Lock Rule | checkbox | Yes | "Prevent the target rule and mutation seed\nfrom being changed by config loads/pastes.\nMutation seed can also be locked independently via Alt-click." | `ui/menu_bar.py:239-242` |

**Help menu** (`ui/menu_bar.py:247`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Guide (toggle) | menu_item | No | — | `ui/menu_bar.py:256-257` |
| Controls (toggle) | menu_item | No | — | `ui/menu_bar.py:258-259` |
| Performance (toggle) | menu_item | No | — | `ui/menu_bar.py:260-261` |
| Parameter Sweeps (toggle) | menu_item | No | — | `ui/menu_bar.py:262-263` |

**Extras menu** (`ui/menu_bar.py:267`):

| Control | Type | Tooltip? | Tooltip Text | Location |
|---|---|---|---|---|
| Config Clipboard | checkbox | Yes | "Set restorable checkpoints with Ctrl-C" | `ui/menu_bar.py:277-281` |
| Screen Recording Controls | checkbox | No | — | `ui/menu_bar.py:284-287` |
| Parameter Locks - EXPERIMENTAL | checkbox | Yes | "Alt-Click on a parameter to freeze it and its value\nwon't change when loading new configs." | `ui/menu_bar.py:290-303` |
| Generics | checkbox | Yes | "8 scratch sliders sent as uniforms to\nentity_update and field_override shaders.\nUseful for live-coding shader experiments." | `ui/menu_bar.py:308-312` |
| Plotting | checkbox | Yes | "GPU histogram visualization from report()\ncalls in entity_update.glsl." | `ui/menu_bar.py:315-319` |
| Radio | checkbox | Yes | "Filter particle visibility by frequency band.\nOnly particles within the target frequency\n+/- bandwidth are visible." | `ui/menu_bar.py:322-326` |
| Scheduled Renders | checkbox | Yes | "Queue multiple render specs for\nunattended batch video rendering." | `ui/menu_bar.py:329-333` |
| Save Simulation State... | menu_item | Yes | "Dump the current entity buffer and 3D canvas\nto disk (particle positions + trail densities)." | `ui/menu_bar.py:338-343` |
| Load Simulation State (submenu, `.fsim` list + per-entry delete button) | begin_menu | No | — | `ui/menu_bar.py:345-376` |

---

## Summary: controls with no tooltip

Quick reference for anyone doing a pass to add missing tooltips, grouped by window:

- **Physics Settings**: all collapsing headers, the "Appearance" menu itself, sweep buttons (X/Y/C), Min/Max fields, Reset buttons in the slider context menu, Mutation Seed text.
- **Preferences**: none missing — every real control has a tooltip.
- **Render Settings**: the entire Camera section (FOV, Aperture, Focal Depth, Move/Rotate Speed, Orbit Center/Rate, Stereogram, Eye Offset, Parallel), most of the Lighting/Sky color & vector fields, Firefly Clamp, Resolution Scale, most OptiX geometry/material/rasterize sub-controls (Sphere Scale, Curve Length/R0/R1, Enable Dish, BRDF, Glossy IOR, Albedo Saturation/Brightness, Ambient, AO toggle/rays/radius, RR Start Depth, Denoise ×2), most OpenGL medium sub-controls (Colored Extinction, Albedo/Extinction RGB & sat/brightness, Density Scale, Enable Dish), all `_persisted_header` collapsing headers, Capture SPP, Re-render Preview.
- **Config Clipboard**: all controls (entry selectable, delete button, rename field).
- **Radio**: all controls (Radio enabled, Target Frequency, Bandwidth).
- **Main Menu**: Load submenu, Preferences toggle, Load Editor Settings submenu, Show/Hide Windows, Reset all slider ranges, Unlock/Lock Everything, all four Help menu toggles, Screen Recording Controls checkbox, Load Simulation State submenu.
