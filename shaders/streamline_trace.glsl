#version 430

// Streamline tracer (stateful, ring-buffered).
//
// Each invocation advances ONE particle by STEPS_PER_DISPATCH integration
// steps and appends each new position to that particle's ring buffer. The
// particle's live position/velocity persist in a separate state buffer, so a
// path is built up across many dispatches instead of being retraced from
// scratch every frame.
//
// This is the structure the audio work needs: dispatches are decoupled from
// the render cadence, and each one deposits a small, fixed number of samples.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Ring of past positions. One slice of RING_CAPACITY vec2 per particle,
// written at (write_index % RING_CAPACITY) and overwriting the oldest entry.
layout(std430, binding = 5) buffer StreamlinePath {
    vec2 path[];
};

// Live integrator state, one record per particle. Kept apart from the ring so
// the history can be cleared or re-read without disturbing the physics.
struct ParticleState {
    vec2 pos;
    vec2 vel;
    uint write_index;  // Total positions ever written (not yet wrapped)
    uint alive;        // 0 = retired (left the canvas)
    uint _pad0;
    uint _pad1;
};

layout(std430, binding = 6) buffer StreamlineState {
    ParticleState particles[];
};

uniform sampler2D canvas_texture;

uniform vec2 seed_pos;              // Cursor / pinned seed, world space [-1, 1]
uniform int STEPS_PER_DISPATCH;     // Integration steps to advance this call
uniform int RING_CAPACITY;          // Ring slots per particle
uniform int STREAMLINE_COUNT;       // Active particles
uniform float FORCE_SCALE;
uniform float DAMPING;
uniform float STEP_SIZE;
uniform float RESTORE_FORCE;
uniform float INITIAL_SPEED;
uniform float RANDOM_SEED;
uniform bool RESPAWN_AT_SEED;       // Retired particles restart at the seed
uniform bool STOP_AT_EDGE;          // Retire on leaving the canvas

vec2 sample_field(vec2 world_pos) {
    return texture(canvas_texture, world_pos * 0.5 + 0.5).xy;
}

bool out_of_bounds(vec2 world_pos) {
    return any(lessThan(world_pos, vec2(-1.0))) ||
           any(greaterThan(world_pos, vec2(1.0)));
}

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

vec2 launch_velocity(int line_id) {
    if (INITIAL_SPEED <= 0.0) {
        return vec2(0.0);
    }
    // Spread the ids far apart before hashing: hash11 of nearly-equal inputs
    // collapses, which would launch every particle identically.
    float h = float(line_id) * 71.13 + RANDOM_SEED * 131.7;
    float angle = hash11(h) * 6.28318530718;
    float speed = mix(0.25, 1.0, hash11(h + 7.77)) * INITIAL_SPEED;
    return vec2(cos(angle), sin(angle)) * speed;
}

void main() {
    int line_id = int(gl_GlobalInvocationID.x);
    if (line_id >= STREAMLINE_COUNT) {
        return;
    }

    int base = line_id * RING_CAPACITY;

    vec2 pos = particles[line_id].pos;
    vec2 vel = particles[line_id].vel;
    uint write_index = particles[line_id].write_index;
    uint alive = particles[line_id].alive;

    // A retired particle either sits out or restarts at the seed.
    if (alive == 0u) {
        if (!RESPAWN_AT_SEED) {
            return;
        }
        pos = seed_pos;
        vel = launch_velocity(line_id);
        alive = 1u;
    }

    for (int k = 0; k < STEPS_PER_DISPATCH; ++k) {
        vec2 force = sample_field(pos);
        vel += force * FORCE_SCALE;

        // Spring toward the seed, applied after the field so its own damping
        // is not scaled by FORCE_SCALE.
        //
        // Semi-implicit and normalised by STEP_SIZE: what matters for
        // stability is the displacement the spring produces per step, which
        // is k * STEP_SIZE. Clamping that product to < 1 keeps a dragged seed
        // pulling the swarm along instead of catapulting it - an unclamped
        // stiff spring turns any seed jump into a huge velocity impulse.
        if (RESTORE_FORCE > 0.0) {
            // ks is the fraction of the gap closed per step. The damping
            // factor below hits zero at ks = 0.25, and past that the swarm
            // slingshots to the canvas corners; measured stable up to ~0.20,
            // so cap at 0.15 for margin and let the slider saturate there.
            float ks = min(STEP_SIZE * STEP_SIZE * RESTORE_FORCE, 0.15);
            vel += (seed_pos - pos) * (ks / max(STEP_SIZE, 1e-6));
            // Critical damping for this discrete step.
            vel *= max(0.0, 1.0 - min(2.0 * sqrt(ks), 1.0));
        }

        vel *= DAMPING;
        //vel += .01*(vec2(-.5+hash11((k+line_id-particles[line_id].write_index)*.007123),hash11((-k+line_id-particles[line_id].write_index)*.06572)));
        pos += vel * STEP_SIZE;

        if (out_of_bounds(pos)) {
            if (STOP_AT_EDGE) {
                alive = 0u;
                break;
            }
            // Otherwise wrap to the opposite edge and keep going. Clamping
            // instead would park the particle against the boundary with the
            // field pushing it outward forever, collapsing its whole tail
            // onto one point. The draw pass detects the resulting position
            // jump and breaks the line strip there.
            pos = fract((pos + 1.0) * 0.5) * 2.0 - 1.0;
        }

        path[base + int(write_index % uint(RING_CAPACITY))] = pos;
        write_index += 1u;
    }

    particles[line_id].pos = pos;
    particles[line_id].vel = vel;
    particles[line_id].write_index = write_index;
    particles[line_id].alive = alive;
}
