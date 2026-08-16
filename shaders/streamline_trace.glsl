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

// The canvas samplers, the physics uniforms and get_can_lerp() all come from
// entity_physics.glsl, spliced in ahead of this file. Streamers evaluate the
// same behaviour as real particles rather than a simplified force model, so
// they must not redeclare any of it.
uniform float FIELD_ALPHA_BASE;   // Blend fraction at the start of this block
uniform float FIELD_ALPHA_STEP;   // Added per integration step

uniform vec2 seed_pos;              // Cursor / pinned seed, entity space
uniform int STEPS_PER_DISPATCH;     // Integration steps to advance this call
uniform int RING_CAPACITY;          // Ring slots per particle
uniform int STREAMLINE_COUNT;       // Active particles
uniform float INITIAL_SPEED;
uniform bool RESPAWN_AT_SEED;       // Retired particles restart at the seed
uniform bool STOP_AT_EDGE;          // Retire on leaving the canvas
uniform float HAZARD_RATE;          // Per-step chance a particle resets to seed
uniform uint DISPATCH_INDEX;        // Decorrelates the hazard roll per dispatch
uniform float SEED_SCATTER;         // Radius of each particle's own spawn disc
uniform uint RUN_SALT;              // Reseeds the whole population per reset

// --- Audio voices ---
uniform bool AUDIO_ENABLED;
uniform int AUDIO_VOICE_COUNT;      // Particles 0..N-1 each drive one voice
uniform int AUDIO_LANE_STRIDE;      // Floats per voice lane (slots * block)
uniform int AUDIO_SLOT_BASE;        // Start index of this block within a lane
uniform float AUDIO_AMPLITUDE;
uniform float AUDIO_HP_COEFF;       // One-pole high-pass coefficient
uniform float AUDIO_RAMP_DEC;       // Per-sample decrement of the reset ramp

// --- Self-trail correction (see self_trail()) ---
uniform float SELF_TRAIL_STRENGTH;     // 0 disables; 1 = full estimated deposit
uniform float SELF_TRAIL_SIZE;         // Brush footprint radius, world units
uniform float SELF_TRAIL_PERSISTENCE;  // Canvas trail persistence
uniform float SELF_TRAIL_SPREAD;       // Kernel widening for canvas diffusion

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

// Trail this particle would have deposited at `at`, had it been a real
// particle at `src` moving at `src_vel`. Mirrors brush.frag:
//   brush_out = vel * (1 - trail_persistence) * gaussian(uv - .5, .163)^2
// over a quad of half-extent `size`, so uv-space radius 0.5 corresponds to
// world distance `size`.
vec2 self_trail(vec2 at, vec2 src, vec2 src_vel) {
    float size = SELF_TRAIL_SIZE;
    if (size <= 0.0) return vec2(0.0);

    // brush.frag deposits vel * (1-persistence) * gaussian(r_uv, .163)^2 over
    // a disc of radius `size`. Two gaussians of sigma s multiply to one of
    // sigma s/sqrt(2), and in world units that is:
    float sigma = size * 0.163 * 2.0 / 1.41421356;

    // The canvas blurs every step, so by the time a sensor reads this ink it
    // has spread well past the original footprint. Without accounting for
    // that the estimate is identically zero at any usable sensor distance -
    // sensors sit 1.7x to 13x the brush radius away. SELF_TRAIL_SPREAD widens
    // the kernel to stand in for that accumulated diffusion.
    sigma *= max(1.0, SELF_TRAIL_SPREAD);

    float d = length(at - src);
    // Total deposited quantity is preserved as the kernel widens, so the
    // peak falls as the spread grows - the same ink over a larger area.
    float peak = (1.0 - SELF_TRAIL_PERSISTENCE)
               / (2.0 * 3.14159265359 * 0.163 * 0.163);
    float g = exp(-(d * d) / (2.0 * sigma * sigma));
    return src_vel * peak * g * SELF_TRAIL_STRENGTH;
}

void main() {
    int line_id = int(gl_GlobalInvocationID.x);
    if (line_id >= STREAMLINE_COUNT) {
        return;
    }

    int base = line_id * RING_CAPACITY;

    // --- Hoisted once per invocation ---
    // The step loop runs up to 512 times and calls calculate_setting ~7 times
    // per step. The PhysicsSettings are uniforms (cheap, and the driver would
    // likely hoist them anyway), but current_rule is 10 FourierCenters read
    // from an SSBO - that one is worth pulling into registers explicitly
    // rather than trusting the optimiser to prove it loop-invariant.
    //
    // Every streamer is cohort 0 for now, which also makes the cohort_sweep
    // and mutation terms constant across the population.
    const float COHORT = 0.0;
    Rule current_rule = get_particle_target_rule();
    if (current_rule.centers[0].frequency == vec4(0)
        && current_rule.centers[5].amplitude == vec4(0)) {
        current_rule = Rule(generate_random_centers(get_particle_rule_seed()));
    }
    mutate_rule(current_rule,
                calculate_setting(get_particle_mutation_scale(), vec2(0), COHORT),
                get_particle_rule_seed());

    PhysicsSetting sensor_distance   = get_particle_sensor_distance();
    PhysicsSetting sensor_angle      = get_particle_sensor_angle();
    PhysicsSetting sensor_gain       = get_particle_sensor_gain();
    PhysicsSetting global_force_mult = get_particle_global_force_mult();
    PhysicsSetting drag              = get_particle_drag();
    PhysicsSetting strafe_power      = get_particle_strafe_power();

    vec2 pos = particles[line_id].pos;
    vec2 vel = particles[line_id].vel;
    // Previous step's start state, for the self-trail correction. Not carried
    // across dispatches: the first step of a block simply goes uncorrected,
    // which is one step in 32-512 and not worth another two state floats.
    vec2 prev_pos = vec2(0);
    vec2 prev_vel = vec2(0);
    bool has_prev = false;
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
        // current one. Read by get_can_lerp() inside the shared physics.
        g_field_alpha = FIELD_ALPHA_BASE + FIELD_ALPHA_STEP * float(k);

        // --- Full Fluoddity particle step, matching entity_update ---
        // Sensor geometry.
        float sample_dist = 1./SQRT_WORLD_SIZE*.005
                          * calculate_setting(sensor_distance, pos, COHORT);
        int orient_mode = get_particle_absolute_orientation();
        float mix_amt = min(1, orient_mode) * ORIENTATION_MIX;
        vec2 orientation = safenorm(vel);
        if (orient_mode == 1) { orientation = mix(orientation, vec2(0,1), mix_amt); }
        else if (orient_mode == 2) { orientation = mix(orientation, -normalize(pos), mix_amt); }
        vec2 left_off = orientation * sample_dist;
        vec2 right_off = orientation * sample_dist;
        float sensor_a = calculate_setting(sensor_angle, pos, COHORT) * PI;
        pR(left_off, sensor_a);
        pR(right_off, -sensor_a);

        // Sensor taps. get_can_lerp blends the previous and current canvas so
        // the field is not piecewise constant across a block of audio samples.
        vec2 ltap = get_can_lerp(pos + left_off);
        vec2 rtap = get_can_lerp(pos + right_off);

        // Self-trail correction.
        //
        // A real particle swims in ink it laid down itself: measured, the
        // canvas at a particle's own position is ~7.8x stronger than at a
        // random point and ~0.78 cosine-aligned with its own velocity. A
        // read-only streamer never deposits, so it sees a hole exactly where
        // a real particle sees its strongest, most self-correlated signal -
        // which is why streamers diverge on configs that lean on that signal.
        //
        // Add back a first-order estimate: the deposit this particle would
        // have made one step ago, evaluated at each sensor. brush.frag lays
        // down vel * (1 - trail_persistence) * gaussian(r)^2 over a disc of
        // radius SELF_TRAIL_SIZE, so this reproduces that at the sensor's
        // distance from the previous position.
        if (SELF_TRAIL_STRENGTH > 0.0 && has_prev) {
            ltap += self_trail(pos + left_off, prev_pos, prev_vel);
            rtap += self_trail(pos + right_off, prev_pos, prev_vel);
        }
        float sensor_scaling = SQRT_WORLD_SIZE * 38.855
                             * calculate_setting(sensor_gain, pos, COHORT);
        ltap *= sensor_scaling;
        rtap *= sensor_scaling;

        // Remember this step's start state; the next step's sensors read the
        // ink that would have been deposited here.
        vec2 step_pos = pos;
        vec2 step_vel = vel;

        vec2 force = vec2(0);
        vec2 strafe = vec2(0);
        vec2 col_params = vec2(0);
        calculate_entity_behavior(ltap, rtap, orientation, current_rule,
                                  pos, COHORT, force, strafe, col_params);

        float gfm = calculate_setting(global_force_mult, pos, COHORT);
        force  *= 1./SQRT_WORLD_SIZE * gfm / 400.;
        strafe *= 1./SQRT_WORLD_SIZE * gfm / 20.;

        // --- Audio sample: instantaneous power ---
        // Taken here, on the force just computed and the velocity it acts on,
        // before drag and the position update. That is the physical work rate
        // at this instant.
        //
        // To try the post-integration variant instead, comment this line out
        // and uncomment the one marked POST-INTEGRATION below, then press V.
        float raw_power = dot(force, vel)*100.;

        // Integrate exactly as a real particle does: velocity IS the step,
        // there is no separate step scale.
        vel = vel * calculate_setting(drag, pos, COHORT) + force;
        pos += vel;
        pos += strafe * calculate_setting(strafe_power, pos, COHORT);

        // POST-INTEGRATION variant: uses the velocity after drag and the
        // force have been applied. Closer to dot(force, force).
        //float raw_power = dot(force, vel);

        prev_pos = step_pos;
        prev_vel = step_vel;
        has_prev = true;

        // Advanced-drawing force/strafe field, same as entity_update.
        vec4 draw_sample = get_field(pos);
        vel += .01 * force_field_strength * draw_sample.xy;
        pos += .01 * strafe_field_strength * draw_sample.zw;

        // Boundary handling. Streamers use the simulation's own boundary mode
        // so they stay inside the same region real particles do.
        float ca_b = canvas_resolution.x / canvas_resolution.y;
        vec2 edge = vec2(sqrt(ca_b), 1.0 / sqrt(ca_b));
        int bmode = get_particle_boundary_conditions();
        if (bmode == 0) {
            if (pos.x < -edge.x || pos.x > edge.x) {
                vel.x = -vel.x;
                pos.x = edgeflect(pos.x / edge.x) * edge.x;
            }
            if (pos.y < -edge.y || pos.y > edge.y) {
                vel.y = -vel.y;
                pos.y = edgeflect(pos.y / edge.y) * edge.y;
            }
        } else if (bmode == 2) {
            // Wrap. The draw pass breaks the line strip at the resulting jump.
            pos = fract((pos + edge) / (2.0 * edge)) * 2.0 * edge - edge;
        } else if (any(greaterThan(abs(pos), edge))) {
            // Reset mode: a streamer that leaves is retired or respawned,
            // rather than jumping to the simulation's reset layout.
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
            pos = clamp(pos, -edge, edge);
        }

        path[base + int(write_index % uint(RING_CAPACITY))] = pos;
        write_index += 1u;

        // --- Audio sample for this integration step ---
        if (is_voice) {
            float raw = raw_power * AUDIO_AMPLITUDE;

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
