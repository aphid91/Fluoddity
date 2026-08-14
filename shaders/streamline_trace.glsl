#version 430

// Streamline tracer.
//
// Each invocation releases one test particle at the cursor and integrates it
// through the canvas vector field for STEPS steps, writing one position per
// step into its own slice of the path buffer.
//
// The particle is treated as having inertia: the canvas vec2 at the
// particle's location is an acceleration, not a velocity. With
// STREAMLINE_COUNT > 1 each particle gets a different random initial
// velocity, so a spray of paths fans out from the same seed.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 5) buffer StreamlinePath {
    // One slice of (MAX_STEPS + 1) vec2 per streamline. Slot 0 of each slice
    // holds that streamline's valid point count (as a float); its points
    // begin at slice offset 1.
    vec2 path[];
};

uniform sampler2D canvas_texture;

uniform vec2 seed_pos;        // Start position in world space [-1, 1]
uniform int STEPS;            // Number of integration steps
uniform int MAX_STEPS;        // Slice stride (buffer capacity per streamline)
uniform int STREAMLINE_COUNT; // Number of streamlines to trace
uniform float FORCE_SCALE;    // Canvas value -> acceleration
uniform float DAMPING;        // Per-step velocity retention
uniform float STEP_SIZE;      // Velocity -> displacement per step
uniform float RESTORE_FORCE;  // Spring pull back toward the seed
uniform float INITIAL_SPEED;  // Magnitude of the random launch velocity
uniform float RANDOM_SEED;    // Reseeds the launch directions each frame

// Sample the field. Canvas is a 2-component (RG) texture: .xy is the
// trail vector field.
vec2 sample_field(vec2 world_pos) {
    vec2 canvas_uv = world_pos * 0.5 + 0.5;
    return texture(canvas_texture, canvas_uv).xy;
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

void main() {
    int line_id = int(gl_GlobalInvocationID.x);
    if (line_id >= STREAMLINE_COUNT) {
        return;
    }

    // Base index of this streamline's slice.
    int base = line_id * (MAX_STEPS + 1);

    int max_steps = min(STEPS, MAX_STEPS);

    vec2 pos = seed_pos;
    vec2 vel = vec2(0.0);

    // A single streamline launches from rest; a spray fans out with random
    // directions so the paths diverge instead of overlapping exactly.
    if (STREAMLINE_COUNT > 1) {
        float angle = hash11(float(line_id) + RANDOM_SEED) * 6.28318530718;
        float speed = hash11(float(line_id) + RANDOM_SEED + 7.77) * INITIAL_SPEED;
        vel = vec2(cos(angle), sin(angle)) * speed;
    }

    // Always record the seed so a stalled particle still renders a point.
    path[base + 1] = pos;
    int written = 1;

    for (int k = 1; k < max_steps; ++k) {
        vec2 force = sample_field(pos);
        force += (seed_pos - pos) * STEP_SIZE * RESTORE_FORCE;
        vel += force * FORCE_SCALE;
        vel *= DAMPING;
        pos += vel * STEP_SIZE;

        // Stop-at-edge: the particle leaves the canvas and the path ends.
        // Remaining slots are left untouched; the draw call is clamped to
        // `written` so stale positions from previous frames never render.
        if (out_of_bounds(pos)) {
            break;
        }

        path[base + k + 1] = pos;
        written = k + 1;
    }

    path[base] = vec2(float(written), 0.0);
}
