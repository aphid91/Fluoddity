#version 430

// Draws the traced streamlines as line strips. There is no vertex buffer:
// each vertex pulls its world position out of the path SSBO by gl_VertexID,
// and gl_InstanceID selects which streamline's slice to read.

layout(std430, binding = 5) buffer StreamlinePath {
    vec2 path[];  // Per slice: [0].x = valid point count, points from index 1
};

uniform vec2 cam_pos;
uniform float cam_zoom;
uniform vec2 canvas_resolution;
uniform vec2 window_size;
uniform int MAX_STEPS;  // Slice stride, must match the trace shader

out float v_t;      // Normalized position along the path, for fading the tail
flat out int v_dead; // Non-zero for vertices past this streamline's valid count

// Inverse of arrow_debug.frag's screen_to_world, so the streamline stays
// registered with the canvas under pan and zoom.
vec2 world_to_ndc(vec2 world_pos) {
    float tex_aspect = canvas_resolution.x / canvas_resolution.y;
    float window_aspect = window_size.x / window_size.y;

    float scale_x, scale_y;
    if (tex_aspect > window_aspect) {
        scale_x = 1.0;
        scale_y = window_aspect / tex_aspect;
    } else {
        scale_x = tex_aspect / window_aspect;
        scale_y = 1.0;
    }

    scale_x /= cam_zoom;
    scale_y /= cam_zoom;

    vec2 ndc;
    ndc.x = world_pos.x * scale_x - cam_pos.x / cam_zoom;
    ndc.y = world_pos.y * scale_y + cam_pos.y / cam_zoom;
    return ndc;
}

void main() {
    int base = gl_InstanceID * (MAX_STEPS + 1);
    int count = int(path[base].x);

    // Streamlines terminate at different steps, but every instance is drawn
    // with the same vertex count. Collapse the surplus vertices onto the last
    // valid point and flag them so the fragment stage discards them; that
    // keeps stale positions from earlier frames from ever being rendered.
    int idx = min(gl_VertexID, max(count - 1, 0));
    v_dead = (gl_VertexID >= count) ? 1 : 0;

    vec2 world_pos = path[base + idx + 1];

    v_t = count > 1 ? float(idx) / float(count - 1) : 0.0;

    gl_Position = vec4(world_to_ndc(world_pos), 0.0, 1.0);
}
