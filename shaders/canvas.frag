#version 430

in vec2 texcoord;

uniform sampler2D can_tex;
out vec2 can_out;

// Draw trail mode uniforms
uniform bool draw_mode;
uniform vec2 mouse;
uniform vec2 previous_mouse;
uniform float draw_size;
uniform float draw_power;

// Advanced drawing uniforms
// brush_mode codes: 0=mouse_dir, 1=inverse, 2=fixed, 3=attract, 4=repel
uniform int brush_mode;
uniform float fixed_direction_heading;
uniform bool erase_mode;       // Right-click eraser active
uniform bool fill_mode;        // Fill entire canvas for one frame
uniform int fill_direction_type; // 0=fixed, 1=radial_in, 2=radial_out
uniform bool canvas_draw_active; // Whether canvas is a draw target

// Boundary conditions
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP

// Tiling mode
uniform bool tiling_mode;

// Canvas dimensions for aspect correction
uniform vec2 canvas_resolution;

// SYNCHRONIZED: This struct must match entity_update.glsl
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag
struct PhysicsSetting {
    float slider_value;
    float min_value;
    float max_value;
    float x_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float y_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float cohort_sweep; // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float jitter;       // 0.0 = off, higher = more randomness (proportional to result)
};

uniform PhysicsSetting TRAIL_PERSISTENCE_SETTING;
uniform PhysicsSetting TRAIL_DIFFUSION_SETTING;

uniform int frame_count;

#define COHORTS 64//TODO THIS IS A HACK, OFTEN WRONG. Cohort sweeps a little broken

// SYNCHRONIZED: This function must match entity_update.glsl and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py
float calculate_setting(PhysicsSetting setting, vec2 pos, float cohort){
    //if no sweep modes are active and no jitter, just return slider value
    if(setting.y_sweep == 0.0 && setting.cohort_sweep == 0.0 && setting.x_sweep == 0.0 && setting.jitter == 0.0)
        {return setting.slider_value;}
    //otherwise calculate parameter sweeps
    pos = (pos+1)/2.;//convert to 0..1 for use as a mix coefficient
    cohort = cohort / COHORTS; //convert to 0..1 for mixing

    // Count active sweeps and accumulate results
    float result = 0;
    int active_sweeps = 0;
    if(setting.x_sweep != 0.0) {
        // For inverse sweep (x_sweep < 0), swap min and max
        if(setting.x_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.x);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.x);
        }
        active_sweeps++;
    }
    if(setting.y_sweep != 0.0) {
        // For inverse sweep (y_sweep < 0), swap min and max
        if(setting.y_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, pos.y);
        } else {
            result += mix(setting.max_value, setting.min_value, pos.y);
        }
        active_sweeps++;
    }
    if(setting.cohort_sweep != 0.0) {
        // For inverse sweep (cohort_sweep < 0), swap min and max
        if(setting.cohort_sweep > 0.0) {
            result += mix(setting.min_value, setting.max_value, cohort);
        } else {
            result += mix(setting.max_value, setting.min_value, cohort);
        }
        active_sweeps++;
    }

    // Average the results or use slider_value if no sweeps
    result = active_sweeps > 0 ? result / float(active_sweeps) : setting.slider_value;

    // Apply jitter: random variation proportional to the result value
    // Uses a simple hash function since canvas.frag doesn't have access to entity_update's hash()
    if(setting.jitter != 0.0) {
        float random = fract(sin(dot(pos + float(frame_count) * 0.01, vec2(12.9898, 78.233))) * 43758.5453) * 2.0 - 1.0;
        result += setting.jitter * result * random;
    }

    return result;
}

vec2 getCan(vec2 p, sampler2D sam) {
    vec2 uv = (BOUNDARY_CONDITIONS_MODE == 2) ? fract(p) : p;
    return texture(sam, uv).rg;
}

vec2 getBlur(vec2 pos, sampler2D sam,float diffusion_constant) {
    ivec2 imsz = textureSize(sam, 0);
    vec3 off = vec3(1. / vec2(imsz), 0);
    vec2 np = pos + off.zy;
    vec2 sp = pos - off.zy;
    vec2 wp = pos - off.xz;
    vec2 ep = pos + off.xz;
    vec2 nc = getCan(np, sam);
    vec2 sc = getCan(sp, sam);
    vec2 wc = getCan(wp, sam);
    vec2 ec = getCan(ep, sam);
    float K = diffusion_constant;
    return (getCan(pos, sam) * K + nc + sc + wc + ec) / (4. + K);
}

// Aspect-correct UV delta so length() is isotropic in entity space
vec2 aspect_correct_uv(vec2 uv_delta) {
    float ca = canvas_resolution.x / canvas_resolution.y;
    return uv_delta * vec2(sqrt(ca), 1.0/sqrt(ca));
}

// Calculate draw vector based on brush mode
vec2 calculate_draw_vector(int mode, vec2 mouse_vel, float heading,
                           vec2 pixel_pos, vec2 mouse_p) {
    if (mode == 0) {
        return mouse_vel;                       // Mouse Direction
    } else if (mode == 1) {
        return -mouse_vel;                      // Inverse Mouse Direction
    } else if (mode == 2) {
        return .01*vec2(sin(heading), cos(heading)); // Fixed Direction (0 = up)
    } else if (mode == 3) {
        // In - Attract (toward mouse)
        vec2 to_mouse = mouse_p - pixel_pos;
        vec2 corrected = aspect_correct_uv(to_mouse);
        float len = length(corrected);
        return len > 0.0 ? .01*corrected / len : vec2(0.0);
    } else if (mode == 4) {
        // Out - Repel (away from mouse)
        vec2 from_mouse = pixel_pos - mouse_p;
        vec2 corrected = aspect_correct_uv(from_mouse);
        float len = length(corrected);
        return len > 0.0 ? .01*corrected / len : vec2(0.0);
    }
    return vec2(0.0);
}

// Gaussian kernel for draw trail mode
float draw_kernel(float distance, float size) {
    // Gaussian: exp(-distance^2 / (2 * sigma^2))
    // Using size as sigma
    float sigma = size;
    return exp(-distance * distance / (2.0 * sigma * sigma));
}

void main() {
    // Clear to zero on frame 0 to prevent garbage data
    if (frame_count == 0) {
        can_out = vec2(0);
        return;
    }

    vec2 can_color;
    // Map texcoord to entity space for parameter sweeps
    float _ca = canvas_resolution.x / canvas_resolution.y;
    vec2 entity_space_pos = (texcoord * 2.0 - 1.0) * vec2(sqrt(_ca), 1.0/sqrt(_ca));
    float TRAIL_DIFFUSION = calculate_setting(TRAIL_DIFFUSION_SETTING,entity_space_pos,0);
    TRAIL_DIFFUSION = clamp(TRAIL_DIFFUSION,0.001,1.0);//keeps jitter from exceeding the valid domain
    if(TRAIL_DIFFUSION>0){
        TRAIL_DIFFUSION= TRAIL_DIFFUSION*TRAIL_DIFFUSION;//better scaling for slider
        TRAIL_DIFFUSION = 4/(pow(5,(TRAIL_DIFFUSION))-1);//better scaling for slider
        can_color = getBlur(texcoord, can_tex,TRAIL_DIFFUSION);
    }
    else{
        can_color = texture(can_tex,texcoord).rg;
    }

    // Use entity space position for position-based sweeps
    float trail_persistence = calculate_setting(TRAIL_PERSISTENCE_SETTING, entity_space_pos, 0.0);
    trail_persistence = clamp(trail_persistence,0.0,0.999);
    can_out = can_color * trail_persistence;

    // Draw trail mode: add velocity based on mouse drag
    if (draw_mode && canvas_draw_active && draw_power > 0.0) {
        float distance_to_mouse;

        // Calculate raw mouse velocity for tiling correction
        vec2 raw_mouse_vel = (mouse - previous_mouse);

        if (tiling_mode) {
            // In tiling mode, check 9-cell neighborhood (3x3) for wrapped distance
            // This allows trail drawing across wrapped edges/corners
            float min_distance = 999.0;
            vec2 min_draw_vector = vec2(999);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    vec2 wrapped_mouse = mouse + vec2(dx, dy);
                    float dist = length(aspect_correct_uv(texcoord - wrapped_mouse));
                    min_distance = min(min_distance, dist);
                    vec2 vel = (wrapped_mouse-previous_mouse);
                    min_draw_vector = length(vel)<length(min_draw_vector)?vel:min_draw_vector;
                }
            }
            distance_to_mouse = min_distance;
            raw_mouse_vel = min_draw_vector;
        } else {
            // Normal mode: direct distance calculation
            distance_to_mouse = length(aspect_correct_uv(texcoord - mouse));
        }

        // Apply brush mode to get final draw vector
        vec2 draw_vector = calculate_draw_vector(
            brush_mode, raw_mouse_vel, fixed_direction_heading,
            texcoord, mouse
        );
        draw_vector *= draw_power/5;

        // Apply Gaussian kernel and add to velocity channels (RG)
        float kernel_weight = draw_kernel(distance_to_mouse, draw_size);
        can_out += draw_vector * kernel_weight/draw_size*(1-trail_persistence);
    }

    // Right-click eraser: hard circle erase within draw_size radius
    if (erase_mode && canvas_draw_active) {
        float erase_distance;
        if (tiling_mode) {
            float min_dist = 999.0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    vec2 wrapped = mouse + vec2(dx, dy);
                    min_dist = min(min_dist, length(aspect_correct_uv(texcoord - wrapped)));
                }
            }
            erase_distance = min_dist;
        } else {
            erase_distance = length(aspect_correct_uv(texcoord - mouse));
        }
        if (erase_distance < draw_size*2) {//Match the reticle size from frame_assembly.frag
            can_out = vec2(0.0);
        }
    }

    // Fill mode: apply brush to entire canvas for one frame (no kernel, no persistence factor)
    if (fill_mode && canvas_draw_active && draw_power > 0.0) {
        vec2 fill_vector;
        if (fill_direction_type == 0) {
            fill_vector = vec2(sin(fixed_direction_heading), cos(fixed_direction_heading));
        } else if (fill_direction_type == 3) {
            // Fixed Direction - Negative (heading + PI)
            float neg_heading = fixed_direction_heading + 3.1415;
            fill_vector = vec2(sin(neg_heading), cos(neg_heading));
        } else if (fill_direction_type == 1) {
            vec2 to_center = vec2(0.5) - texcoord;
            vec2 corrected = aspect_correct_uv(to_center);
            float len = length(corrected);
            fill_vector = len > 0.0 ? corrected / len : vec2(0.0);
        } else {
            vec2 from_center = texcoord - vec2(0.5);
            vec2 corrected = aspect_correct_uv(from_center);
            float len = length(corrected);
            fill_vector = len > 0.0 ? corrected / len : vec2(0.0);
        }
        fill_vector *= draw_power / 5.0;
        can_out += .25*fill_vector;
    }
}
