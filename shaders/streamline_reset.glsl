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
    uint _pad0;
    uint _pad1;
};

layout(std430, binding = 6) buffer StreamlineState {
    ParticleState particles[];
};

uniform vec2 seed_pos;
uniform int RING_CAPACITY;
uniform int STREAMLINE_COUNT;
uniform float INITIAL_SPEED;
uniform float RANDOM_SEED;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

void main() {
    int line_id = int(gl_GlobalInvocationID.x);
    if (line_id >= STREAMLINE_COUNT) {
        return;
    }

    vec2 vel = vec2(0.0);
    if (INITIAL_SPEED > 0.0) {
        // Spread the ids far apart before hashing: hash11 of nearly-equal
        // inputs collapses, which would launch every particle identically.
        float h = float(line_id) * 71.13 + RANDOM_SEED * 131.7;
        float angle = hash11(h) * 6.28318530718;
        float speed = mix(0.25, 1.0, hash11(h + 7.77)) * INITIAL_SPEED;
        vel = vec2(cos(angle), sin(angle)) * speed;
    }

    particles[line_id].pos = seed_pos;
    particles[line_id].vel = vel;
    particles[line_id].write_index = 0u;
    particles[line_id].alive = 1u;

    // Collapse the ring onto the seed so no stale geometry survives the reset.
    int base = line_id * RING_CAPACITY;
    for (int i = 0; i < RING_CAPACITY; ++i) {
        path[base + i] = seed_pos;
    }
}
