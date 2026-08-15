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
    // Audio voice state, carried across dispatches like pos/vel. Packed into
    // the existing padding so the struct stays 32 bytes.
    float hp_x1;       // Previous raw sample   (one-pole high pass)
    float hp_y1;       // Previous filtered out (one-pole high pass)
    float ramp;        // Post-reset gain ramp, counts down 1 -> 0
    float _pad0;
};

layout(std430, binding = 6) buffer StreamlineState {
    ParticleState particles[];
};

// Per-voice audio lanes. Each contributing particle owns a lane of
// AUDIO_LANE_STRIDE floats; sample k of the current block for voice v lives
// at v * AUDIO_LANE_STRIDE + AUDIO_SLOT_BASE + k. A separate reduce pass
// sums the lanes down to the mix, so no atomics are needed and the
// summation order is deterministic.
layout(std430, binding = 7) buffer StreamlineAudio {
    float audio_samples[];
};

uniform sampler2D canvas_texture;
// The canvas as it was at the previous physics step. The canvas only updates
// every physics step, which is dozens to hundreds of audio samples apart, so
// sampling it directly makes the field piecewise constant and the audio
// staircases audibly. Blending toward the current frame across the samples
// between steps removes that.
uniform sampler2D canvas_prev_texture;
uniform bool FIELD_INTERPOLATE;
uniform float FIELD_ALPHA_BASE;   // Blend fraction at the start of this block
uniform float FIELD_ALPHA_STEP;   // Added per integration step

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
uniform float HAZARD_RATE;          // Per-step chance a particle resets to seed
uniform uint DISPATCH_INDEX;        // Decorrelates the hazard roll per dispatch
uniform float SEED_SCATTER;         // Radius of each particle's own spring target
uniform uint RUN_SALT;              // Reseeds the whole population per reset

// --- Audio voices ---
uniform bool AUDIO_ENABLED;
uniform int AUDIO_VOICE_COUNT;      // Particles 0..N-1 each drive one voice
uniform int AUDIO_LANE_STRIDE;      // Floats per voice lane (slots * block)
uniform int AUDIO_SLOT_BASE;        // Start index of this block within a lane
uniform float AUDIO_AMPLITUDE;
uniform float AUDIO_HP_COEFF;       // One-pole high-pass coefficient
uniform float AUDIO_RAMP_DEC;       // Per-sample decrement of the reset ramp

vec2 sample_field(vec2 world_pos) {
    return texture(canvas_texture, world_pos * 0.5 + 0.5).xy;
}

// Field at a point in time between the previous physics step and the current
// one. alpha 0 = previous frame, 1 = current.
//
// The two textures only ever describe the most recent physics interval, so
// alpha is clamped rather than wrapped. A dispatch that outruns that interval
// holds at the current frame instead of replaying the same prev->cur sweep,
// which would be a sawtooth - a different artifact rather than a fix. Keeping
// the block short enough to sit inside one physics step is what makes the
// interpolation cover the whole block (see AUDIO_MAX_STEPS_PER_PHYSICS).
vec2 sample_field_lerp(vec2 world_pos, float alpha) {
    vec2 uv = world_pos * 0.5 + 0.5;
    vec2 cur = texture(canvas_texture, uv).xy;
    if (!FIELD_INTERPOLATE) {
        return cur;
    }
    vec2 prev = texture(canvas_prev_texture, uv).xy;
    return mix(prev, cur, clamp(alpha, 0.0, 1.0));
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

// Integer bit-mix (PCG-style). Used for the per-step hazard roll: hash11 on
// nearly-equal float inputs aliases badly, which would make whole batches of
// particles respawn on the same step instead of independently.
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

// Launch velocity from an explicit 32-bit stream id, so every respawn draws
// fresh randomness. Deriving it from line_id alone (as before) handed a
// respawning particle the same velocity every time, and it would retrace an
// identical path.
vec2 launch_velocity(int line_id, uint stream) {
    if (INITIAL_SPEED <= 0.0) {
        return vec2(0.0);
    }
    float angle = rand01(uint(line_id) * 2u + 0u, stream) * 6.28318530718;
    float speed = mix(0.25, 1.0, rand01(uint(line_id) * 2u + 1u, stream))
                * INITIAL_SPEED;
    return vec2(cos(angle), sin(angle)) * speed;
}

// A stable per-particle offset from the seed. The spring is a point
// attractor: without this every particle converges on exactly the same fixed
// point and the population collapses to a single dot. Giving each its own
// target turns that collapse into a cloud around the seed.
vec2 seed_offset(int line_id) {
    if (SEED_SCATTER <= 0.0) {
        return vec2(0.0);
    }
    float angle = rand01(uint(line_id) * 2u + 0u, 0x5eed0001u) * 6.28318530718;
    // sqrt keeps the samples uniform over the disc instead of bunching at
    // the centre.
    float r = sqrt(rand01(uint(line_id) * 2u + 1u, 0x5eed0001u)) * SEED_SCATTER;
    return vec2(cos(angle), sin(angle)) * r;
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

    // Audio voice state. Carried across dispatches so the filter and the
    // post-reset ramp survive block boundaries.
    // The first AUDIO_VOICE_COUNT particles each drive a voice.
    bool is_voice = AUDIO_ENABLED && line_id < AUDIO_VOICE_COUNT;
    int lane = line_id * AUDIO_LANE_STRIDE + AUDIO_SLOT_BASE;
    float hp_x1 = particles[line_id].hp_x1;
    float hp_y1 = particles[line_id].hp_y1;
    float ramp = particles[line_id].ramp;

    // A retired particle either sits out or restarts at the seed.
    if (alive == 0u) {
        if (!RESPAWN_AT_SEED) {
            // A voice must still fill its whole lane even when its particle
            // is retired, or the mix would read stale samples from the
            // previous time this slot was written.
            if (is_voice) {
                for (int k = 0; k < STEPS_PER_DISPATCH; ++k) {
                    // Let the filter relax toward zero rather than freezing a
                    // DC offset in place.
                    hp_y1 *= AUDIO_HP_COEFF;
                    audio_samples[lane + k] = hp_y1;
                }
                particles[line_id].hp_y1 = hp_y1;
            }
            return;
        }
        pos = seed_pos + seed_offset(line_id);
        vel = launch_velocity(line_id, RUN_SALT ^ (DISPATCH_INDEX * 0x9e3779b9u));
        alive = 1u;
        ramp = 1.0;
    }

    for (int k = 0; k < STEPS_PER_DISPATCH; ++k) {
        // Hazard: an independent per-step chance of respawning at the seed.
        // Defined per step rather than per dispatch so the rate means the
        // same thing whatever STEPS_PER_DISPATCH and the dispatch rate are.
        if (HAZARD_RATE > 0.0) {
            uint roll_id = RUN_SALT ^ (DISPATCH_INDEX + uint(k));
            if (rand01(uint(line_id), roll_id) < HAZARD_RATE) {
                pos = seed_pos + seed_offset(line_id);
                // Fresh stream per respawn, so a particle that respawns
                // repeatedly does not retrace the same path each time.
                vel = launch_velocity(line_id, roll_id * 0x85ebca6bu + 1u);
                // No extra ring write here: the position jump back to the
                // seed is what the draw pass keys on to break the strip, and
                // writing the seed twice would just burn a ring slot.
                //
                // The audio voice carries its filter state across the reset
                // (so the DC step decays instead of jumping) and re-arms the
                // gain ramp, which hides the discontinuity in dot(vel, field).
                ramp = 1.0;
            }
        }

        // Where this step sits between the previous physics frame and the
        // current one.
        float field_alpha = FIELD_ALPHA_BASE + FIELD_ALPHA_STEP * float(k);
        vec2 force = sample_field_lerp(pos, field_alpha);
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
            // Each particle pulls toward its own offset target, so a stiff
            // spring gathers the population into a cloud rather than
            // collapsing every particle onto one identical fixed point.
            vec2 target = seed_pos + seed_offset(line_id);
            vel += (target - pos) * (ks / max(STEP_SIZE, 1e-6));
            // Critical damping for this discrete step.
            vel *= max(0.0, 1.0 - min(2.0 * sqrt(ks), 1.0));
        }

        vel *= DAMPING;
        //vel += .01*(vec2(-.5+hash11((k+line_id-particles[line_id].write_index)*.007123),hash11((-k+line_id-particles[line_id].write_index)*.06572)));
        pos += vel * STEP_SIZE;

        if (out_of_bounds(pos)) {
            if (STOP_AT_EDGE) {
                alive = 0u;
                // Pad the rest of the block so it stays exactly
                // STEPS_PER_DISPATCH samples long; a short block would
                // desynchronise the audio stream.
                if (is_voice) {
                    for (int j = k; j < STEPS_PER_DISPATCH; ++j) {
                        hp_y1 *= AUDIO_HP_COEFF;
                        audio_samples[lane + j] = hp_y1;
                    }
                }
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

        // --- Audio sample for this integration step ---
        if (is_voice) {
            // Project the particle's motion onto the field it is moving
            // through: large when it is being driven hard, near zero when it
            // drifts across a null.
            // Same interpolated field the step used, so the sample and the
            // motion that produced it agree.
            float raw = dot(vel, sample_field_lerp(pos, field_alpha))
                      * AUDIO_AMPLITUDE;

            // One-pole high pass (DC blocker). Carrying x1/y1 across a reset
            // turns the position discontinuity into a decaying step rather
            // than a permanent offset.
            float y = AUDIO_HP_COEFF * (hp_y1 + raw - hp_x1);
            hp_x1 = raw;
            hp_y1 = y;

            // Post-reset gain ramp. The high pass fixes the DC step but not
            // the instantaneous jump, which would otherwise click.
            if (ramp > 0.0) {
                y *= (1.0 - ramp);
                ramp = max(0.0, ramp - AUDIO_RAMP_DEC);
            }

            audio_samples[lane + k] = y;
        }
    }

    particles[line_id].pos = pos;
    particles[line_id].vel = vel;
    particles[line_id].write_index = write_index;
    particles[line_id].alive = alive;
    particles[line_id].hp_x1 = hp_x1;
    particles[line_id].hp_y1 = hp_y1;
    particles[line_id].ramp = ramp;
}
