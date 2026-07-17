"""Renderer-specific offline video strategies (Step 7, Phase D).

Moves the two renderer-specific video paths out of ``simulation_runner`` and
behind the ``VideoStrategy`` contract:

- ``TracerVideoStrategy``   — volumetric tracer (progressive SPP + re-splat motion blur).
- ``OptixPtVideoStrategy``  — OptiX path tracer offline (substep motion blur).

Both share a ``VideoContext`` carrying the sim / camera / controller-cam / window
and a ``run_physics_step`` callback (still owned by ``SimulationRunner``). Each
strategy owns its own per-output-frame accumulation state and returns a finished,
tonemapped texture from ``run_frame`` when an output frame is complete (else None).
The behavior is a faithful move of the original methods — no logic change.
"""
from __future__ import annotations

from dataclasses import dataclass

import glfw
import moderngl
import numpy as np

# NOTE: ``services.stereogram`` is imported lazily inside the methods that use
# it. A top-level import here would run services/__init__ while the rendering
# package is still initializing (rendering is imported early by camera.py),
# triggering a circular import (services -> config_saver -> ui -> services).


# Minimal fullscreen-quad copy shader used to composite the two stereogram eye
# textures side-by-side into one HDR frame. Plain texture sample, no tonemap.
_COMPOSITE_VS = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""
_COMPOSITE_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
"""


class _StereoCompositor:
    """Composites two half-width HDR eye textures into one full-width frame.

    Owns a full-width rgba16f FBO and a trivial copy program. Each eye texture
    is drawn into its screen half by restricting the GL viewport.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._prog = ctx.program(vertex_shader=_COMPOSITE_VS,
                                 fragment_shader=_COMPOSITE_FS)
        quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype='f4')
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.vertex_array(
            self._prog, [(self._vbo, '2f 2f', 'in_pos', 'in_uv')])
        self._tex = None
        self._fbo = None
        self._size = (0, 0)

    def _ensure(self, width, height):
        if self._size == (width, height):
            return
        if self._tex is not None:
            self._tex.release()
        if self._fbo is not None:
            self._fbo.release()
        self._tex = self.ctx.texture((width, height), 4, dtype='f2')
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._fbo = self.ctx.framebuffer([self._tex])
        self._size = (width, height)

    def begin(self, full_width, height):
        """Bind the composite FBO and clear it. Blit each eye, then finish()."""
        self._ensure(full_width, height)
        self._full = (full_width, height)
        self._fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

    def blit_half(self, tex, viewport):
        """Draw one eye texture into its screen-half viewport of the composite.

        Call between begin() and finish(). The eye texture is sampled fully
        across the given viewport (a half-width sub-rect of the full frame).
        """
        self._fbo.use()
        self.ctx.viewport = viewport
        tex.use(location=0)
        self._prog['u_tex'].value = 0
        self._vao.render()

    def finish(self):
        """Restore the full viewport and return the composited HDR texture."""
        self.ctx.viewport = (0, 0, self._full[0], self._full[1])
        return self._tex


@dataclass
class VideoContext:
    """Shared plumbing a video strategy needs to drive physics + camera."""
    sim: object
    camera: object
    controller_cam: object
    window: object
    run_physics_step: object  # callable(ui_state, step_index)

    def compute_view_proj(self):
        cam = self.controller_cam
        width, height = glfw.get_framebuffer_size(self.window)
        aspect = width / max(height, 1)
        return self.camera.compute_fps_view_proj(
            cam.pos, cam.dir, cam.up, cam.fov, aspect
        )


class TracerVideoStrategy:
    """Volumetric tracer video: progressive SPP with re-splat motion blur."""

    def __init__(self, ctx: VideoContext, tracer_interface):
        self.ctx = ctx
        self.ti = tracer_interface
        self.init_frame_state()

    def init_frame_state(self):
        self._frame_started = False
        self._physics_steps_done = 0
        self._slot_index = 0            # Next cadence slot to sample.
        self._slot_samples_done = 0     # Samples accumulated at current slot.
        self._plan = None               # CapturePlan for this output frame.

    def run_frame(self, ui_state):
        from .capture_cadence import build_capture_plan

        c = self.ctx
        ti = self.ti
        rec = ui_state.preferences.recording

        # --- Start a new output frame if needed ---
        if not self._frame_started:
            # Cadence-lock plan shared with the OptiX backend. Capture SPP is
            # authoritative for the total sample count; Blur Quality only sets
            # which physics frames are eligible to be sampled.
            self._plan = build_capture_plan(
                physics_rate=rec.motion_blur_samples,
                capture_spp=ti.num_samples,
                blur_quality=rec.recording_blur_quality,
                motion_blur=rec.recording_motion_blur,
            )
            self._slot_index = 0
            self._slot_samples_done = 0
            self._physics_steps_done = 0

            # Advance physics to the first cadence slot.
            first_slot = self._plan.slots[0]
            while self._physics_steps_done < first_slot.physics_before:
                c.run_physics_step(ui_state, self._physics_steps_done)
                self._physics_steps_done += 1

            view_proj = c.compute_view_proj()
            width, height = glfw.get_framebuffer_size(c.window)
            scale = max(0.1, ti.resolution_scale)
            rt_width = max(1, int(width * scale))
            rt_height = max(1, int(height * scale))
            cam_right, cam_up = c.camera.compute_fps_camera_basis(
                c.controller_cam.dir, c.controller_cam.up
            )

            # Stereogram: build per-eye (view_proj, region) at render scale. Both
            # eyes accumulate into disjoint half-regions of the same target.
            from services import stereogram
            stereo_eyes = None
            stereo = stereogram.params_from_camera_state(ui_state.camera)
            if stereo.enabled:
                cam = c.controller_cam
                stereo_eyes = []
                for side in stereogram.eye_sides():
                    ev = stereogram.eye_camera(
                        cam.pos, cam.dir, cam.up, cam.fov,
                        rt_width, rt_height, side, stereo)
                    eye_vp = c.camera.compute_fps_view_proj(
                        ev.pos, ev.dir, ev.up, ev.fov, ev.aspect)
                    stereo_eyes.append((eye_vp, ev.viewport))

            ti.start_video_render(
                c.sim.get_entity_buffer(), c.sim.entity_count,
                view_proj, rt_width, rt_height,
                camera_right=cam_right, camera_up=cam_up,
                stereo_eyes=stereo_eyes,
            )
            self._frame_started = True
            # Physics is already at the first slot; sampling begins there.

        # --- Advance physics to the current cadence slot (if not already) ---
        slot = self._plan.slots[self._slot_index]
        while self._physics_steps_done < slot.physics_before:
            c.run_physics_step(ui_state, self._physics_steps_done)
            self._physics_steps_done += 1
            # Re-splat entities with updated positions (keeps accumulation)
            view_proj = c.compute_view_proj()
            ti.re_splat(c.sim.get_entity_buffer(), c.sim.entity_count, view_proj)

        # --- Is this the final sample of the output frame? If so, run out any
        # remaining physics first so the frame advances exactly physics_rate
        # steps (playback-speed lock) — tick_video resolves on the last sample. ---
        is_last_slot = (self._slot_index == len(self._plan.slots) - 1)
        is_last_sample = (is_last_slot
                          and self._slot_samples_done == slot.samples - 1)
        if is_last_sample:
            while self._physics_steps_done < self._plan.total_physics:
                c.run_physics_step(ui_state, self._physics_steps_done)
                self._physics_steps_done += 1
                view_proj = c.compute_view_proj()
                ti.re_splat(c.sim.get_entity_buffer(), c.sim.entity_count,
                            view_proj)

        # --- Accumulate 1 SPP at this slot ---
        frame_complete = ti.tick_video()
        self._slot_samples_done += 1
        if self._slot_samples_done >= slot.samples:
            # Move to the next cadence slot on the following calls.
            self._slot_index += 1
            self._slot_samples_done = 0

        if frame_complete:
            # Match the shared tonemap curve used by the other backends.
            ti.brightness = ui_state.preferences.rendering.brightness
            ti.tonemap_softness = ui_state.preferences.rendering.tonemap_softness
            display_tex = ti.tonemap_for_video()
            self._frame_started = False
            return display_tex
        return None


class OptixPtVideoStrategy:
    """OptiX path tracer offline video: substep motion blur."""

    def __init__(self, ctx: VideoContext, pt_interface):
        self.ctx = ctx
        self.pt = pt_interface
        self._compositor = None
        self.init_frame_state()

    def init_frame_state(self):
        self._frame_started = False
        self._slot_index = 0          # Next cadence slot to sample this frame.
        self._physics_steps_done = 0  # Physics steps run so far this frame.
        self._plan = None             # CapturePlan for the current output frame.
        # Per-output-frame stereo plan: list of eye dicts, or None (mono).
        self._eyes = None
        self._full_width = 0
        self._render_height = 0

    def _build_eyes(self, ui_state):
        """Return a list of eye descriptors for this output frame.

        Mono: one eye with accum_slot 0 and full-width dims. Stereo: two eyes,
        each with its offset camera, accum_slot (1/2), half-width render dims,
        and the screen-half viewport (in render pixels) for compositing.
        """
        from services import stereogram

        c = self.ctx
        cam = c.controller_cam
        stereo = stereogram.params_from_camera_state(ui_state.camera)

        fb_w, fb_h = glfw.get_framebuffer_size(c.window)
        scale = max(0.1, ui_state.preferences.rendering.render_resolution_scale)
        full_w = max(1, int(fb_w * scale))
        height = max(1, int(fb_h * scale))
        self._full_width = full_w
        self._render_height = height

        if not stereo.enabled:
            return [dict(pos=cam.pos, dir=cam.dir, up=cam.up, fov=cam.fov,
                         slot=0, width=full_w, height=height,
                         viewport=(0, 0, full_w, height))]

        eyes = []
        for eye_index, side in enumerate(stereogram.eye_sides()):
            ev = stereogram.eye_camera(
                cam.pos, cam.dir, cam.up, cam.fov, full_w, height, side, stereo)
            vp = ev.viewport
            eyes.append(dict(pos=ev.pos, dir=ev.dir, up=ev.up, fov=ev.fov,
                             slot=eye_index + 1,
                             width=vp[2], height=vp[3], viewport=vp))
        return eyes

    def _advance_physics_to(self, ui_state, target_steps):
        """Run physics steps until this frame has advanced ``target_steps``."""
        while self._physics_steps_done < target_steps:
            self.ctx.run_physics_step(ui_state, self._physics_steps_done)
            self._physics_steps_done += 1

    def _accumulate_slot(self, slot):
        """Trace ``slot.samples`` SPP for every eye at the current particle
        state, in one offline_substep call per eye (single GAS build). The
        renderer normalizes by the actual accumulated sample count, so a
        variable number of samples per slot integrates correctly."""
        for eye in self._eyes:
            self.pt.offline_substep(
                eye['pos'], eye['dir'], eye['up'], eye['fov'],
                accum_slot=eye['slot'], spp=slot.samples)

    def run_frame(self, ui_state):
        from .capture_cadence import build_capture_plan

        c = self.ctx
        pt = self.pt
        rec = ui_state.preferences.recording
        plan = build_capture_plan(
            physics_rate=rec.motion_blur_samples,
            capture_spp=ui_state.preferences.rendering.capture_spp,
            blur_quality=rec.recording_blur_quality,
            motion_blur=rec.recording_motion_blur,
        )

        # --- Start a new output frame if needed ---
        if not self._frame_started:
            self._plan = plan
            self._physics_steps_done = 0

            first_slot = plan.slots[0]
            # Advance physics to the first cadence slot so entities have moved.
            self._advance_physics_to(ui_state, first_slot.physics_before)

            # Lock in the eye plan (mono or stereo) for this whole output frame.
            self._eyes = self._build_eyes(ui_state)

            # Begin an offline accumulation per eye (each in its own slot). We
            # drive sampling one SPP at a time, so spp_per_substep is 1; the
            # total sample count is governed by the CapturePlan, not by
            # total_substeps * spp_per_substep.
            for eye in self._eyes:
                pt.start_offline_render(
                    entity_buffer=c.sim.get_entity_buffer(),
                    entity_count=c.sim.entity_count,
                    width=eye['width'],
                    height=eye['height'],
                    total_substeps=len(plan.slots),
                    spp_per_substep=1,
                    accum_slot=eye['slot'],
                )
            self._accumulate_slot(first_slot)

            self._frame_started = True
            self._slot_index = 1
            return None

        # --- Subsequent cadence slots: advance physics, then accumulate ---
        if self._slot_index < len(self._plan.slots):
            slot = self._plan.slots[self._slot_index]
            self._advance_physics_to(ui_state, slot.physics_before)
            self._accumulate_slot(slot)
            self._slot_index += 1
            return None

        # --- All slots sampled: run out the remaining physics so this output
        # frame always advances exactly physics_rate steps (playback-speed lock),
        # then finish each eye and (if stereo) composite. ---
        self._advance_physics_to(ui_state, self._plan.total_physics)
        eyes = self._eyes
        self._frame_started = False

        if len(eyes) == 1:
            return pt.finish_offline_render(flip_y=False, accum_slot=0)

        # Stereo: resolve each eye to HDR and blit its half into the composite
        # immediately — finish_offline_render returns the renderer's SHARED
        # scratch texture, so the next eye's resolve would overwrite it.
        if self._compositor is None:
            self._compositor = _StereoCompositor(self.ctx.camera.ctx)
        self._compositor.begin(self._full_width, self._render_height)
        for idx, eye in enumerate(eyes):
            last = (idx == len(eyes) - 1)
            tex = pt.finish_offline_render(
                flip_y=False, accum_slot=eye['slot'], end_session=last)
            if tex is None:
                return None
            self._compositor.blit_half(tex, eye['viewport'])
        return self._compositor.finish()
