#version 430

// Resets particle state: places every particle at the seed with a fresh
// launch velocity and empties its ring. Runs on demand (the Reset button, a
// count change, or the first dispatch), so the CPU never uploads state.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 5) buffer StreamlinePath {
    vec2 path[];
};

struct ParticleState {
    vec2 pos;
    vec2 vel;
    uint write_index;
    uint alive;
    float hp_x1;   // Audio one-pole high-pass state
    float hp_y1;
    float ramp;    // Post-reset gain ramp
    float _pad0;
};

layout(std430, binding = 6) buffer StreamlineState {
    ParticleState particles[];
};

uniform vec2 seed_pos;
uniform int RING_CAPACITY;
uniform int STREAMLINE_COUNT;
uniform float INITIAL_SPEED;
uniform float SEED_SCATTER;
uniform uint RUN_SALT;

// Same integer bit-mix as the tracer, so a reset and a respawn draw from the
// same well-distributed stream.
uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(uint a, uint b) {
    return float(hash_u32(a * 0x9e3779b9u ^ hash_u32(b))) * (1.0 / 4294967296.0);
}

void main() {
    int line_id = int(gl_GlobalInvocationID.x);
    if (line_id >= STREAMLINE_COUNT) {
        return;
    }

    vec2 vel = vec2(0.0);
    if (INITIAL_SPEED > 0.0) {
        float angle = rand01(uint(line_id) * 2u + 0u, RUN_SALT) * 6.28318530718;
        float speed = mix(0.25, 1.0, rand01(uint(line_id) * 2u + 1u, RUN_SALT))
                    * INITIAL_SPEED;
        vel = vec2(cos(angle), sin(angle)) * speed;
    }

    // Match the tracer's seed_offset exactly, so a particle starts at the
    // same target the spring will pull it toward.
    vec2 start = seed_pos;
    if (SEED_SCATTER > 0.0) {
        float a = rand01(uint(line_id) * 2u + 0u, 0x5eed0001u) * 6.28318530718;
        float r = sqrt(rand01(uint(line_id) * 2u + 1u, 0x5eed0001u)) * SEED_SCATTER;
        start += vec2(cos(a), sin(a)) * r;
    }

    particles[line_id].pos = start;
    particles[line_id].vel = vel;
    particles[line_id].write_index = 0u;
    particles[line_id].alive = 1u;
    // A full reset is a hard restart, so clear the filter rather than
    // carrying it: there is no previous lifetime to blend out of.
    particles[line_id].hp_x1 = 0.0;
    particles[line_id].hp_y1 = 0.0;
    particles[line_id].ramp = 1.0;  // Still ramp in, to avoid a click

    // Collapse the ring onto the start point so no stale geometry survives.
    int base = line_id * RING_CAPACITY;
    for (int i = 0; i < RING_CAPACITY; ++i) {
        path[base + i] = start;
    }
}
