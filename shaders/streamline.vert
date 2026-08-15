#version 430

// Draws each particle's recent history as a line strip. There is no vertex
// buffer: vertices are pulled from the ring by gl_VertexID, and gl_InstanceID
// selects the particle.
//
// Vertex 0 is the OLDEST point of the visible tail and the last vertex is the
// newest, so the head/tail fade runs the right way along the path.

layout(std430, binding = 5) buffer StreamlinePath {
    vec2 path[];
};

struct ParticleState {
    vec2 pos;
    vec2 vel;
    uint write_index;
    uint alive;
    uint _pad0;
    uint _pad1;
};

layout(std430, binding = 6) buffer StreamlineState {
    ParticleState particles[];
};

uniform vec2 cam_pos;
uniform float cam_zoom;
uniform vec2 canvas_resolution;
uniform vec2 window_size;
uniform int RING_CAPACITY;
uniform int TAIL_LENGTH;      // Vertices drawn per instance
uniform float JUMP_THRESHOLD; // Gap that counts as a teleport, not motion

out float v_t;
flat out int v_dead;

// Inverse of arrow_debug.frag's screen_to_world, so the path stays registered
// with the canvas under pan and zoom.
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
    int line_id = gl_InstanceID;
    int base = line_id * RING_CAPACITY;

    uint written = particles[line_id].write_index;

    // Only as much history as exists, capped by the ring and the tail slider.
    int available = int(min(written, uint(RING_CAPACITY)));
    int tail = min(TAIL_LENGTH, available);

    // Walk back from the newest sample. age 0 = newest.
    int age = (tail - 1) - gl_VertexID;
    v_dead = (gl_VertexID >= tail || age < 0) ? 1 : 0;
    age = max(age, 0);

    // Newest entry sits one slot behind the write cursor.
    int newest = int((written + uint(RING_CAPACITY) - 1u) % uint(RING_CAPACITY));
    int slot = newest - age;
    slot -= RING_CAPACITY * int(floor(float(slot) / float(RING_CAPACITY)));

    vec2 world_pos = path[base + slot];

    // A boundary wrap or a hazard respawn teleports the particle. Drop the
    // vertex that follows such a jump so the strip breaks instead of drawing
    // a line straight across the canvas. JUMP_THRESHOLD is derived from the
    // integrator's real step scale, so genuine motion never trips it.
    int prev_slot = slot - 1;
    prev_slot -= RING_CAPACITY * int(floor(float(prev_slot) / float(RING_CAPACITY)));
    if (age < tail - 1 &&
        distance(world_pos, path[base + prev_slot]) > JUMP_THRESHOLD) {
        v_dead = 1;
    }

    // 0 at the oldest visible point, 1 at the newest.
    v_t = tail > 1 ? 1.0 - float(age) / float(tail - 1) : 1.0;

    gl_Position = vec4(world_to_ndc(world_pos), 0.0, 1.0);
}
