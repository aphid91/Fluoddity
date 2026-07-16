#version 450
layout(local_size_x = 64) in;
#define LINEAR_MC_DEPOSIT

//SAME STRUCT USED IN CAM_BRUSH.VERT AND POINTS_3D.VERT
struct Entity {
    float px, py, pz;    // position (3D)
    float vx, vy, vz;    // velocity (3D)
    float hue;
    float size;
};  // Total: 32 bytes (8 floats)

// Entity accessors — keep physics code readable despite explicit-float layout
vec3 get_pos(Entity e) { return vec3(e.px, e.py, e.pz); }
vec3 get_vel(Entity e) { return vec3(e.vx, e.vy, e.vz); }
void set_pos(inout Entity e, vec3 p) { e.px = p.x; e.py = p.y; e.pz = p.z; }
void set_vel(inout Entity e, vec3 v) { e.vx = v.x; e.vy = v.y; e.vz = v.z; }
struct Rule {
    FourierCenter centers[10];
};
layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};
layout(std430, binding = 2) buffer RuleBuffer {
    Rule rules[];
};
// SYNCHRONIZED: This struct must match canvas_update_3d.glsl
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas_update_3d.glsl
struct PhysicsSetting {
    float slider_value;
    float min_value;
    float max_value;
    float x_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float y_sweep;      // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float cohort_sweep; // 0.0 = off, 1.0 = normal sweep, -1.0 = inverse sweep
    float jitter;       // 0.0 = off, higher = more randomness (proportional to result)
};
// Canvas-relative scaling: values that were proportional to sqrt(world_size)
// now scale as canvas_dim / 256 (the original default resolution).
#define CANVAS_DIM_DEFAULT 256.0
#define CANVAS_SCALE (float(canvas_3d_size.x) / CANVAS_DIM_DEFAULT)
uniform int frame_count;
uniform Rule target_rule;
uniform sampler3D canvas_3d; //trails canvas (RGBA16F packed: R=vx, G=vy, B=vz)
uniform ivec3 canvas_3d_size;  // (W, H, D) for non-cube support
uniform vec2 canvas_resolution;
uniform PhysicsSetting DRAG_SETTING; 
uniform PhysicsSetting STRAFE_POWER_SETTING;
uniform PhysicsSetting SENSOR_ANGLE_SETTING;
uniform PhysicsSetting GLOBAL_FORCE_MULT_SETTING;
uniform PhysicsSetting SENSOR_DISTANCE_SETTING;
uniform PhysicsSetting AXIAL_FORCE_SETTING;
uniform PhysicsSetting LATERAL_FORCE_SETTING;
uniform PhysicsSetting SENSOR_GAIN_SETTING;
uniform PhysicsSetting MUTATION_SCALE_SETTING;
uniform PhysicsSetting HAZARD_RATE_SETTING;
uniform PhysicsSetting TRAIL_PERSISTENCE_SETTING;
layout(rgba16f, binding = 0) uniform image3D can_img;
uniform float HUE_SENSITIVITY;
uniform bool COLOR_BY_COHORT;
uniform bool DISABLE_SYMMETRY;
const bool TESTING_MODE = false;
const int PLANE_SAMPLES = 1;
uniform int ABSOLUTE_ORIENTATION; // 0=Off, 1=Y axis, 2=Radial
uniform float ORIENTATION_MIX; // Blend factor for orientation calculations
uniform float GRAVITY_FORCE;  // Gravity-like force [-1,1]
uniform float GRAVITY_STRAFE; // Gravity-like strafe [-1,1]
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP
uniform int RESET_MODE; //0-1-2-3 == FLAT_GRID-RANDOM-RING-GRID_3D
uniform float INIT_SPACING; //0..1 spacing multiplier for reset() (1=default)
uniform int COHORTS; //each cohort gets its own rule and starting location
uniform float RULE_SEED;
uniform bool WRITE_RULES; // Set true for one frame when rule buffer readback is needed
uniform vec4 generic03;
uniform vec4 generic47;

// "Enable Dish" toggle from the active renderer: gates the SDF collider block.
uniform int enable_collider;      // 0=Off, non-zero = collide with the dish SDF

// Radio feature uniforms
uniform int RADIO_ENABLED;        // 0=Off, >0 = enabled mode
uniform float RADIO_TARGET_FREQ;  // Target frequency
uniform float RADIO_BANDWIDTH;    // Bandwidth

// Histogram reporting SSBO
layout(std430, binding = 5) buffer ReportsBuffer {
    uvec4 reports[];
};
uniform vec4 hist_min;
uniform vec4 hist_max;
uniform uint bucket_count;
uniform uvec4 plot_mode;

void report(float val, uint plot_num) {
    uint ch = plot_num % 4u;
    if (plot_mode[ch] == 0u) return;
    float lo = hist_min[ch];
    float hi = hist_max[ch];
    float t = clamp((val - lo) / (hi - lo), 0.0, 1.0);
    uint bucket_idx = min(uint(t * float(bucket_count)), bucket_count - 1u);
    atomicAdd(reports[bucket_idx][ch], 1u);
}

////////////////////////////CONSTANTS
#define PI 3.1415926
// ACTIVE_COUNT is now injected by prepend_defines() alongside ENTITY_COUNT and RULE_BUFFER_SIZE

// 6D noise channel wiring (edit and hot-reload with V key)
// INPUT_PROJECTION true:  sensor taps projected to 2D (u,v). false: 3D (u,v,w), filling fourier inputs 5-6
// OUTPUT_PROJECTION true: 4 fourier outputs -> 2D force/strafe. false: 6 outputs -> 3D force/strafe
#define INPUT_PROJECTION false
#define OUTPUT_PROJECTION false
                            //Entities with index > ACTIVE_COUNT aren't rendered or updated
int get_particle_cohorts() {
    return COHORTS;
}
//Calculate the actual setting value for this particle. When sweeps are
//active, physics settings can depend on entity position and cohort
// SYNCHRONIZED: This function must match canvas_update_3d.glsl and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas_update_3d.glsl, sim.py
float calculate_setting(PhysicsSetting setting, vec2 pos, float cohort){
    //if no sweep modes are active and no jitter, just return slider value
    if(setting.y_sweep == 0.0 && setting.cohort_sweep == 0.0 && setting.x_sweep == 0.0 && setting.jitter == 0.0)
        {return setting.slider_value;}
    //otherwise calculate parameter sweeps
    pos = (pos+1)/2.;//convert to 0..1 for use as a mix coefficient
    cohort = floor(cohort) / float(get_particle_cohorts()); //convert to 0..1 for mixing

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
    // hash() returns 0..1, so (hash(...)*2.-1.) returns -1..1
    if(setting.jitter != 0.0) {
        float random = hash(vec2(float(frame_count)+result, pos.x + pos.y * 1000.0)) * 2.0 - 1.0;
        result += setting.jitter * result * random;
    }

    return result;
}



PhysicsSetting get_particle_axial_force() {
    return AXIAL_FORCE_SETTING;
}

PhysicsSetting get_particle_lateral_force() {
    return LATERAL_FORCE_SETTING;
}

PhysicsSetting get_particle_sensor_gain() {
    return SENSOR_GAIN_SETTING;
}

PhysicsSetting get_particle_mutation_scale() {
    return MUTATION_SCALE_SETTING;
}

PhysicsSetting get_particle_drag() {
    return DRAG_SETTING;
}

PhysicsSetting get_particle_strafe_power() {
    return STRAFE_POWER_SETTING;
}

PhysicsSetting get_particle_sensor_angle() {
    return SENSOR_ANGLE_SETTING;
}

PhysicsSetting get_particle_global_force_mult() {
    return GLOBAL_FORCE_MULT_SETTING;
}

PhysicsSetting get_particle_sensor_distance() {
    return SENSOR_DISTANCE_SETTING;
}

// Symmetry mode: 0=off, 1=dorsoventral, 2=sagittal (bilateral)
// Currently driven by DISABLE_SYMMETRY bool; true->0, false->2
int get_symmetry_mode() {
    return DISABLE_SYMMETRY ? 0 : 1;
}
int get_particle_symmetry_mode() {
    return get_symmetry_mode();
}

int get_particle_absolute_orientation() {
    return ABSOLUTE_ORIENTATION;
}
PhysicsSetting get_particle_hazard_rate() {
    return HAZARD_RATE_SETTING;
}

int get_particle_boundary_conditions() {
    return BOUNDARY_CONDITIONS_MODE;
}

int get_particle_reset_mode() {
    return RESET_MODE;
}

float get_particle_init_spacing() {
    return INIT_SPACING;
}

float get_particle_hue_sensitivity() {
    return HUE_SENSITIVITY;
}

bool get_particle_color_by_cohort() {
    return COLOR_BY_COHORT;
}

float get_particle_rule_seed() {
    return RULE_SEED;
}

Rule get_particle_target_rule() {
    return target_rule;
}


////////////////////////////////////
//FOURIER NOISE IS IMPORTED INTO THIS SHADER
//FROM fourier6_6.glsl
////////////////////////////////////

//rotate p around origin by angle a
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}


//convert p (entity space, 3D) to texture coords and retrieve 3-component canvas
vec3 get_can_3d(vec3 p){
    float ca = float(canvas_3d_size.x) / float(canvas_3d_size.y);
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv_xy = p.xy / (2.0 * half_extent) + 0.5;
    // Z maps from [-1,1] to [0,1] (or [-z_edge,z_edge] once we have proper 3D bounds)
    float uv_z = p.z * 0.5 + 0.5;
    if(get_particle_boundary_conditions() == 2) {
        uv_xy = fract(uv_xy);
        uv_z = fract(uv_z);
    }
    // When canvas_3d_size.z == 1, uv_z is clamped to center of single slice
    if(canvas_3d_size.z <= 1) uv_z = 0.5;
    vec3 uvw = vec3(uv_xy, uv_z);
    return texture(canvas_3d, uvw).rgb;
}
// DISABLED — legacy 2D force/strafe field sampler. Retained (stubbed) for a
// future field reimplementation; the live drawing runtime was removed in the
// 3D-only cleanup. Returns zero so the (also-disabled) callers stay inert.
vec4 get_field(vec2 p){
    return vec4(0);
    //if(!advanced_drawing_resources_initialized)return vec4(0);
    //vec2 res=textureSize(field_texture,0);
    //float ca = res.x / res.y;
    //vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    //vec2 uv = p / (2.0 * half_extent) + 0.5;
    //if(get_particle_boundary_conditions() == 2) uv = fract(uv);
    //return texture(field_texture, uv);
}

// Gravity-like force expansion: maps a linear -1..1 slider (GRAVITY_FORCE /
// GRAVITY_STRAFE) to a logarithmic physical force so a small knob covers a wide
// range. Odd-symmetric, with a linear dead-zone near center so it reaches 0.
// SYNCHRONIZED: constants + curve must match scripts/migrate_legacy_gravity.py
//   physical = sign(c) * MAXV * 10^(DECADES*(|c|-1))         for |c| > KNEE
//   physical = sign(c) * V_KNEE * (|c|/KNEE)                 for |c| <= KNEE
#define GRAVITY_MAXV    0.5   // physical value at |control| = 1
#define GRAVITY_DECADES 4.0   // log span: MAXV .. MAXV/10^DECADES
#define GRAVITY_KNEE    0.05  // |control| below this ramps linearly to 0
float gravity_expand(float c){
    float a = abs(c);
    float s = sign(c);
    float v_knee = GRAVITY_MAXV * pow(10.0, GRAVITY_DECADES*(GRAVITY_KNEE - 1.0));
    if (a <= GRAVITY_KNEE) {
        return s * v_knee * (a / GRAVITY_KNEE);
    }
    return s * GRAVITY_MAXV * pow(10.0, GRAVITY_DECADES*(a - 1.0));
}

vec2 safenorm(vec2 p){
    return length(p)==0?vec2(0):normalize(p);
}
vec3 safenorm3(vec3 p){
    return length(p)<=1e-10?vec3(0):normalize(p);
}

void build_tangent_plane(
    vec3 vel,
    float theta,
    out vec3 u,
    out vec3 v
) {
    // Forward direction
    u = normalize(vel);

    // Pick something not parallel to u
    vec3 arbitrary =
        abs(u.x) < 0.9
        ? vec3(1,0,0)
        : vec3(0,1,0);

    // First perpendicular direction
    vec3 t1 = normalize(cross(u, arbitrary));

    // Second perpendicular direction
    vec3 t2 = cross(u, t1);

    // Random perpendicular vector around u
    v = cos(theta) * t1 + sin(theta) * t2;
}


float get_cohort(uint index) {
    return float(get_particle_cohorts()) * float(index) / float(ACTIVE_COUNT);
}

//Return all entities to their initialization state
void reset(uint index){

    float size=index<ACTIVE_COUNT?.00015/CANVAS_SCALE: 0;//The main update will set the size once we are "initialized"
    //
    float cohort_val = get_cohort(index);
    float aspect = sqrt(canvas_resolution.x/canvas_resolution.y);

    //set pos and vel to random values on a small ball (3D)
    float cohort_scale = 0.019;//Size of each cluster
    vec3 pos=cohort_scale*vec3(hash(vec2(cohort_val)),hash(vec2(cohort_val+index+2.142)),hash(vec2(cohort_val+index+7.531)));
    vec3 vel=.00005*(vec3(hash(vec2(cohort_val,index)),hash(vec2(cohort_val,pos.y)),hash(vec2(index,pos.z+3.77)))*2-1);

    //RESET_MODE: 0=Flat Grid, 1=Random, 2=Ring/Sphere, 3=3d Grid
    int reset_mode = get_particle_reset_mode();
    int cohorts = get_particle_cohorts();
    // Spacing (0..1): for grids, shrinks the distance between cohort cells while
    // keeping each glob (cell_radius) the same size. At 0 all cells collapse to
    // the origin (replaces the old CRUNCH define). For Random/Ring it is applied
    // below as a plain multiplier on the final position.
    float init_spacing = get_particle_init_spacing();
    if(reset_mode == 0) {
        //FLAT GRID: position cohorts in a 2D grid on the XZ plane, Y near the bottom
        int grid_side = int(ceil(sqrt(float(cohorts))));
        int total_slots = grid_side * grid_side;
        int offset = (total_slots - cohorts) / 2;
        int slot = int(cohort_val) + offset;
        int gx = slot % grid_side;
        int gz = slot / grid_side;
        // Grid cell center in [-0.9, 0.9] for X and Z, Y near bottom.
        // Spacing scales the XZ spread and pulls Y toward center; at spacing 0
        // every cell shares the origin.
        vec3 cell_center = vec3(
            1.8 * ((float(gx) + 0.5) / float(grid_side) - 0.5),
            -0.85,
            1.8 * ((float(gz) + 0.5) / float(grid_side) - 0.5)
        );
        cell_center.xz *= init_spacing;
        // Rejection-sample a disk in XZ, thin spread in Y
        float cell_radius = 0.09 / float(grid_side) / CANVAS_SCALE;
        vec2 candidate_xz;
        float seed_offset = 0.0;
        for (int attempt = 0; attempt < 16; attempt++) {
            candidate_xz = vec2(
                hash(vec2(cohort_val + index, 1.0 + seed_offset)),
                hash(vec2(cohort_val + index, 2.0 + seed_offset))
            ) * 2.0 - 1.0;
            if (dot(candidate_xz, candidate_xz) <= 1.0) break;
            seed_offset += 2.0;
        }
        float candidate_y = (hash(vec2(cohort_val + index, 3.0 + seed_offset)) * 2.0 - 1.0) * 0.1;
        pos = cell_center + vec3(candidate_xz.x, candidate_y, candidate_xz.y) * cell_radius;
    }
    else if(reset_mode == 3) {
        //3D GRID: position cohorts in a centered 3D grid (next-largest cube with gaps)
        int grid_side = int(ceil(pow(float(cohorts), 1.0/3.0)));
        int total_slots = grid_side * grid_side * grid_side;
        // Center the filled slots within the cube: skip (total_slots - cohorts)/2 at the start
        int offset = (total_slots - cohorts) / 2;
        int slot = int(cohort_val) + offset;
        int gx = slot % grid_side;
        int gy = (slot / grid_side) % grid_side;
        int gz = slot / (grid_side * grid_side);
        // Grid cell center in [-0.9, 0.9], scaled toward origin by spacing
        vec3 cell_center = 1.8 * ((vec3(gx, gy, gz) + 0.5) / float(grid_side) - 0.5);
        cell_center *= init_spacing;
        // Rejection-sample a sphere inscribed in the grid cell for isotropic distribution
        float cell_radius = 0.09 / float(grid_side)/CANVAS_SCALE;
        vec3 candidate;
        float seed_offset = 0.0;
        for (int attempt = 0; attempt < 16; attempt++) {
            candidate = vec3(
                hash(vec2(cohort_val + index, 1.0 + seed_offset)),
                hash(vec2(cohort_val + index, 2.0 + seed_offset)),
                hash(vec2(cohort_val + index, 3.0 + seed_offset))
            ) * 2.0 - 1.0;
            if (dot(candidate, candidate) <= 1.0) break;
            seed_offset += 3.0;
        }
        pos = cell_center + candidate * cell_radius;
    }
    else if(reset_mode == 1) {
        //RANDOM: rejection-sample from the sphere inscribing the unit cube
        vec3 candidate;
        float seed_offset = 0.0;
        for (int attempt = 0; attempt < 16; attempt++) {
            candidate = vec3(
                hash(vec2(cohort_val, 1.0 + seed_offset)),
                hash(vec2(cohort_val, 2.0 + seed_offset)),
                hash(vec2(cohort_val, 3.0 + seed_offset))
            ) * 2.0 - 1.0;
            if (dot(candidate, candidate) <= 1.0) break;
            seed_offset += 3.0;
        }
        pos = candidate * init_spacing;
    }
    else if(reset_mode == 2) {
        //SPHERE: arrange cohorts on a spherical shell
        float phi = hash(vec2(cohort_val, 4.0)) * 2.0 * PI;
        float cos_theta = hash(vec2(cohort_val, 5.0)) * 2.0 - 1.0;
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        float radius = 0.5;
        pos = .5*vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta) * radius * init_spacing;
    }
    float hue = 0;
    if(get_particle_color_by_cohort()) {hue = hash(vec2(floor(cohort_val)));}
    //pos = nearest_surf(pos);
    //store to persistent entity buffer
    entities[index]=Entity(pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, hue, size);
}

//randomly change noise function parameters, scaled by parameter amount. 
//Each cohort gets a unique mutation for any given rule
void mutate_rule(inout Rule current_rule,float amount,float cohort){
    float seed = hash(current_rule.centers[4].frequency.xy+current_rule.centers[7].amplitude.yx+current_rule.centers[1].frequency.zw)+cohort;

    for(int i = 0; i < 10; i++) {
        vec4 amp_mutation = amount * (-1.0 + 2.0 * hash4(-.5+vec2(-i+seed,i)));
        current_rule.centers[i].amplitude += amp_mutation;
        current_rule.centers[i].frequency *= 1 + amount * 0.5 * (hash(vec2(seed,i))-.5);

        // Mutate extension fields (dims 5-6)
        vec2 amp_ext_mutation = amount * (-1.0 + 2.0 * vec2(
            hash(vec2(seed + 100.0, float(i))),
            hash(vec2(seed + 200.0, float(i)))
        ));
        current_rule.centers[i].amplitude_ext += amp_ext_mutation;
        current_rule.centers[i].frequency_ext *= 1 + amount * 0.5 * (hash(vec2(seed + 300.0, float(i))) - .5);
    }
}


//Used to enforce left-right symmetry in the local coordinates vec2(forward, left)
vec2 y_reflect(vec2 p){
    return p*vec2(1,-1);
}
vec3 w_reflect(vec3 p){ return vec3(p.x, p.y, -p.z); } // negate w (the chiral axis)
vec2 x_reflect(vec2 p){
    return p*vec2(-1,1);
}
//reflect across the boundary [-1,1] to keep particle positions from leaving the canvas
float edgeflect(float x){
    return sign(x)*(1-abs(1-abs(x)));
}

//Somewhat arbitrary generator of functions with 4 float inputs and 4 float outputs,
//varying rule should smoothly change the behavior of the function
vec4 black_box(vec2 L,vec2 R,Rule rule){
    return (fourier_noise(rule.centers, vec4(L,R), vec2(0.0)));
}

//6-output version: 6 float inputs (L.xyz, R.xyz) -> 6 float outputs (result_lo.xyzw, result_hi.xy)
void black_box_6(vec3 L,vec3 R,Rule rule, out vec4 result_lo, out vec2 result_hi){
    fourier_noise_6(rule.centers, vec4(L.xy, R.xy), vec2(L.z, R.z), result_lo, result_hi);
}



//This function determines entity output by plugging sensor values into a noise function called black_box()
//The calculation is performed twice, once in mirrored coordinates, and the two values are averaged.
//This keeps entities from displaying clockwise/counterclockwise bias.
//PARAMETERS:
//--L and R: velocity field measurements from left sensor and right sensor (vec3: xy=plane, z=normal).
//    When INPUT_PROJECTION is true, L.z and R.z are 0 (caller zeros them).
//--axis: forward vector that defines our orientation (2D in the tangent plane).
//--rule: coefficients for the noise function that dictates entity behavior.
//--pos: entity position (for parameter sweeps)
//--cohort: entity cohort (for parameter sweeps)
//RETURNS:
//--force: A "push" vector in tangent-plane coords (xy=plane, z=normal). z=0 when OUTPUT_PROJECTION is true.
//--strafe: A "hop" vector in tangent-plane coords. z=0 when OUTPUT_PROJECTION is true.
//--color: vec2 to be used as parameters in a coloring function
void calculate_entity_behavior(vec3 L, vec3 R, vec2 axis, Rule rule, vec2 pos, float cohort, out vec3 force, out vec3 strafe, out vec2 color){

    //build a local coordinate frame where "axis" is forward.
    vec2 forward=safenorm(axis);
    vec2 left=vec2(forward.y,-forward.x);

    //Convert L and R xy to local coordinates; z (normal component) passes through unchanged
    L=vec3(dot(L.xy,forward),dot(L.xy,left),L.z);
    R=vec3(dot(R.xy,forward),dot(R.xy,left),R.z);

#if !OUTPUT_PROJECTION
    //6D noise path: use all 6 inputs and 6 outputs
    vec4 base_lo, mirror_lo;
    vec2 base_hi, mirror_hi;
    black_box_6(L, R, rule, base_lo, base_hi);

    int sym = get_particle_symmetry_mode();
    if (sym == 0) {
        // No symmetry — skip mirror pass
        mirror_lo = vec4(0); mirror_hi = vec2(0);
    } else if (sym == 1) {
        // Dorsoventral: w-reflect, no L/R swap
        black_box_6(w_reflect(L), w_reflect(R), rule, mirror_lo, mirror_hi);
    } else {
        // Sagittal (bilateral): w-reflect + swap L<->R
        black_box_6(w_reflect(R), w_reflect(L), rule, mirror_lo, mirror_hi);
    }

    //Combine base and mirror terms: u,v even; w odd
    force = vec3(
        base_lo.x + mirror_lo.x,   // u: even
        base_lo.y + mirror_lo.y,   // v: even
        base_hi.x - mirror_hi.x    // w: odd
    );
    strafe = vec3(
        base_lo.z + mirror_lo.z,   // u: even
        base_lo.w + mirror_lo.w,   // v: even
        base_hi.y - mirror_hi.y    // w: odd
    );

    //Convert force and strafe xy back to world coordinates; z scaled separately
    float axial_s = calculate_setting(get_particle_axial_force(),pos,cohort);
    float lateral_s = calculate_setting(get_particle_lateral_force(),pos,cohort);
    force = vec3(forward*force.x*axial_s + left*force.y*lateral_s, force.z*axial_s);
    strafe = vec3(forward*strafe.x*axial_s + left*strafe.y*lateral_s, strafe.z*axial_s);

    color = base_lo.xy + mirror_lo.xy;
#else
    //Original 4D noise path (L.z and R.z are ignored)
    vec4 baseterm= black_box(L.xy,R.xy,rule);
    vec4 mirrorterm=black_box(y_reflect(R.xy),y_reflect(L.xy),rule);
    if(get_particle_symmetry_mode() == 0){mirrorterm = vec4(0);}

    //Combine base and mirror terms
    vec2 f2 = baseterm.xy+y_reflect(mirrorterm.xy);
    vec2 s2 = baseterm.zw + y_reflect(mirrorterm.zw);

    //Convert force and strafe back to world coordinates
    f2=forward*f2.x*calculate_setting(get_particle_axial_force(),pos,cohort)+left*f2.y*calculate_setting(get_particle_lateral_force(),pos,cohort);
    s2 = forward*s2.x*calculate_setting(get_particle_axial_force(),pos,cohort) + left * s2.y * calculate_setting(get_particle_lateral_force(),pos,cohort);

    force = vec3(f2, 0.0);
    strafe = vec3(s2, 0.0);
    color = baseterm.xy+(mirrorterm.xy);
#endif
    return;
}
// Run one sample of the 2D physics projected onto a random tangent plane.
// Returns force and strafe in 3D world coordinates.
void sample_plane_physics(
    inout vec3 force_accum, inout vec3 strafe_accum, inout vec2 col_accum,
    vec3 pos, vec3 vel, Rule current_rule, float cohort,
    int sample_index, vec2 epos2
) {
    vec3 vel_dir;
    if (length(vel) > 1e-10) {
        vel_dir = normalize(vel);
    } else {
        // Deterministic random direction per-particle to avoid axis bias
        float idx_f = float(gl_GlobalInvocationID.x);
        float phi = hash(vec2(idx_f, 1.0)) * 2.0 * PI;
        float cos_theta = hash(vec2(idx_f, 2.0)) * 2.0 - 1.0;
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        vel_dir = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    }

    vec3 u, v; // Tangent plane basis vectors

    if (TESTING_MODE) {
        // Always use XY plane: u = (1,0,0), v = (0,1,0)
        u = vec3(1,0,0);
        v = vec3(0,1,0);
    } else {

        
        vec3 env = (get_can_3d(pos));
        env = safenorm3(env);
        u = vel_dir;
        v = (cross(u,cross(u,env)));
        if(length(v)>1e-10){
            v = normalize(v);
        }
        else{
            // Random angle for this sample
            float r0 =  hash(vec2(
                float(frame_count) + float(gl_GlobalInvocationID.x) / float(ACTIVE_COUNT),
                float(sample_index)
            ));
            float theta =r0 * 2.0 * PI;
            build_tangent_plane(vel_dir, theta, u, v);
        }

    }

    // Calculate sensor distance
    float sample_dist = 1./CANVAS_SCALE*.005 * calculate_setting(get_particle_sensor_distance(), epos2, cohort);

    // In 3D, the tangent plane is perpendicular to vel_dir, so projecting
    // velocity onto it yields zero. Instead, use u as forward direction —
    // theta already randomizes which direction u points within the plane.
    vec2 orientation;
    if (TESTING_MODE) {
        // 2D mode: velocity lies in the XY plane, project normally
        orientation = safenorm(vec2(dot(vel, u), dot(vel, v)));
    } else {
        // 3D mode: use tangent plane basis directly (randomized by theta)
        orientation = vec2(1, 0);
    }

    // Absolute orientation modes (project reference directions onto plane)
    int ORIENTATION_MODE = get_particle_absolute_orientation();
    float mix_amt = min(1, ORIENTATION_MODE) * ORIENTATION_MIX;
    if (ORIENTATION_MODE == 1) {
        // Y-axis orientation: project (0,1,0) onto the plane
        vec2 y_axis_on_plane = vec2(dot(vec3(0,1,0), u), dot(vec3(0,1,0), v));
        orientation = mix(orientation, safenorm(y_axis_on_plane), mix_amt);
    } else if (ORIENTATION_MODE == 2) {
        // Radial orientation: project -normalize(pos) onto the plane
        vec3 radial = -safenorm3(pos);
        vec2 radial_on_plane = vec2(dot(radial, u), dot(radial, v));
        orientation = mix(orientation, safenorm(radial_on_plane), mix_amt);
    }

    // Sensor offsets in 2D plane coordinates, then lifted to 3D
    vec2 left_offset_2d = orientation * sample_dist;
    vec2 right_offset_2d = orientation * sample_dist;
    float sensor_angle = calculate_setting(get_particle_sensor_angle(), epos2, cohort) * PI;
    pR(left_offset_2d, sensor_angle);
    pR(right_offset_2d, -sensor_angle);

    // Convert 2D plane offsets to 3D world offsets
    vec3 left_offset_3d = left_offset_2d.x * u + left_offset_2d.y * v;
    vec3 right_offset_3d = right_offset_2d.x * u + right_offset_2d.y * v;

    // Read 3D canvas at sensor positions
    vec3 ltap_3d = get_can_3d(pos + left_offset_3d);
    vec3 rtap_3d = get_can_3d(pos + right_offset_3d);

    // Normal vector for the tangent plane
    vec3 w = cross(u, v);

    // Rescale sensor values
    float sensor_scaling = CANVAS_SCALE*38.855 * calculate_setting(get_particle_sensor_gain(), epos2, cohort);

    // Global force multiplier for output rescaling
    float gfm = 1./CANVAS_SCALE * calculate_setting(get_particle_global_force_mult(), epos2, cohort);

    // Project 3D trail vectors onto the tangent plane (+ normal when INPUT_PROJECTION is false)
#if !INPUT_PROJECTION
    vec3 ltap = vec3(dot(ltap_3d, u), dot(ltap_3d, v), dot(ltap_3d, w));
    vec3 rtap = vec3(dot(rtap_3d, u), dot(rtap_3d, v), dot(rtap_3d, w));
#else
    vec3 ltap = vec3(dot(ltap_3d, u), dot(ltap_3d, v), 0.0);
    vec3 rtap = vec3(dot(rtap_3d, u), dot(rtap_3d, v), 0.0);
#endif
    ltap *= sensor_scaling;
    rtap *= sensor_scaling;

    // Run physics on the tangent plane
    vec3 force_local = vec3(0);
    vec3 strafe_local = vec3(0);
    vec2 col_params = vec2(0);
    calculate_entity_behavior(ltap, rtap, orientation, current_rule, epos2, cohort, force_local, strafe_local, col_params);
    if(gl_GlobalInvocationID%500==0){report(ltap.x,0);report(ltap.y,1);report(ltap.z,2);}

    // Rescale output forces
    force_local *= gfm / 400.;
    strafe_local *= gfm / 20.;

    // Lift tangent-plane coords to 3D world coordinates
    vec3 force_3d = force_local.x * u + force_local.y * v + force_local.z * w;
    vec3 strafe_3d = strafe_local.x * u + strafe_local.y * v + strafe_local.z * w;

    force_accum += force_3d;
    strafe_accum += strafe_3d;
    col_accum += col_params;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ENTITY_COUNT) return;

    // Inactive entities get zeroed out. Position offscreen so they don't accidentally get clicked on
    if (index >= ACTIVE_COUNT) {
        entities[index] = Entity(10000.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }
    Entity e=entities[index];
    float cohort = get_cohort(index);

    Rule current_rule=get_particle_target_rule();
    //if a few arbitrary coefficients are exactly 0, then assume target_rule is all 0s (no target) and generate a random rule instead.
    if(current_rule.centers[0].frequency==vec4(0) && current_rule.centers[5].amplitude==vec4(0)){
        current_rule = Rule(generate_random_centers(get_particle_rule_seed()+floor(cohort)));
    }
    //Each cohort gets a random mutation
    mutate_rule(current_rule,calculate_setting(get_particle_mutation_scale(),vec2(e.px,e.py),cohort),get_particle_rule_seed()+floor(cohort));
    // Only write rules when explicitly requested.
    // Rule buffer is fixed at RULE_BUFFER_SIZE entries; each cohort maps to one slot.
    // Only the first entity per cohort writes (cohort boundary detection).
    int cohort_slot = int(floor(cohort)) % RULE_BUFFER_SIZE;
    if(WRITE_RULES) {
        bool is_first_in_cohort = (index == 0u) ||
            (floor(cohort) != floor(float(get_particle_cohorts()) * float(index - 1u) / float(ACTIVE_COUNT)));
        if(is_first_in_cohort) {
            rules[cohort_slot] = current_rule;
        }
    }
    
    //frame_count == 0 signals a simulation reset
    vec2 epos2 = vec2(e.px, e.py); // 2D position for parameter sweeps
    if (frame_count==0||calculate_setting(get_particle_hazard_rate(),epos2,cohort)>hash(vec2(float(index)/float(ACTIVE_COUNT),frame_count)))
    {reset(index);return;}



    //Monte Carlo plane sampling: sample PLANE_SAMPLES random tangent planes,
    //run 2D physics on each, and average the resulting 3D forces.
    vec3 pos3 = get_pos(e);
    vec3 vel3 = get_vel(e);
    vec3 force_accum = vec3(0);
    vec3 strafe_accum = vec3(0);
    vec2 col_accum = vec2(0);
    int num_samples = max(1, PLANE_SAMPLES);
    for (int s = 0; s < num_samples; s++) {
        sample_plane_physics(force_accum, strafe_accum, col_accum,
                             pos3, vel3, current_rule, cohort, s, epos2);
    }
    vec3 force3 = force_accum / float(num_samples);
    vec3 strafe3 = strafe_accum / float(num_samples);
    vec2 col_params = col_accum / float(num_samples);

    //if(index%500==0){report(length(col_params),0);}//small sample

    //Set entity hue (saturation/brightness/alpha are computed in vertex shaders)
    e.hue = abs(get_particle_hue_sensitivity()*col_params.x);
    if(abs(col_params.x-generic03.x*5)<2*generic03.y){e.hue=-e.hue;}
    float negrmin2 = 1./(.5*.5);
    float negrmax2 = 1./(1.5*1.5);
    float szscal = 1./sqrt(negrmin2-hash(vec2(index))*(negrmin2-negrmax2));//exp(-hash(vec2(index))*50);//abs(col_params.y);//hash(vec2());
    e.size = 0.00015/CANVAS_SCALE*szscal;
    //INVISIBILITY RADIO FEATURE
    if(RADIO_ENABLED>0){
        if(!(abs(col_params.x-RADIO_TARGET_FREQ)<RADIO_BANDWIDTH)){e.size=.0;}
    }
    if(get_particle_color_by_cohort()) {e.hue = hash(vec2(floor(cohort)));}

    //Accelerate: Apply drag and add force to e.vel (now 3D)
    float drag = calculate_setting(get_particle_drag(),epos2,cohort);
    e.vx = e.vx*drag + force3.x;
    e.vy = e.vy*drag + force3.y;
    e.vz = e.vz*drag + force3.z;

    //Move: add e.vel and strafe to e.pos (now 3D)
    float strafe_power = calculate_setting(get_particle_strafe_power(),epos2,cohort);
    e.px += e.vx + strafe3.x*strafe_power;
    e.py += e.vy + strafe3.y*strafe_power;
    e.pz += e.vz + strafe3.z*strafe_power;

    //TESTING_MODE: clamp z to 0
    if (TESTING_MODE) {
        e.pz = 0.0;
        e.vz = 0.0;
    }
    if(enable_collider != 0){
        vec4 bonk = collider(vec3(e.px,e.py,e.pz))-vec4(.01,0,0,0);
        if(bonk.x<0){
            vec3 n = bonk.x*-bonk.yzw*.91;
            e.px+=n.x;
            e.py+=n.y;
            e.pz+=n.z;
        }
    }
    // DISABLED — legacy 2D ADVANCED DRAWING force / strafe (applied to XY only).
    // Retained (commented out) for a future field reimplementation; the live
    // drawing runtime + its uniforms were removed in the 3D-only cleanup.
    //vec4 draw_sample =get_field(vec2(e.px, e.py));
    //e.vy += .01/CANVAS_SCALE*force_field_strength*draw_sample.y;
    //e.py += .01/CANVAS_SCALE*strafe_field_strength*draw_sample.w;
    //vec3 sp = vec3(e.px,e.py,e.pz);
    // GRAVITY_FORCE / GRAVITY_STRAFE are linear -1..1 sliders; expand to a
    // logarithmic physical force before applying (see gravity_expand()).
    float force_field_strength = -gravity_expand(GRAVITY_FORCE);
    float strafe_field_strength = -gravity_expand(GRAVITY_STRAFE);
    e.vy += .01/CANVAS_SCALE*force_field_strength;
    e.py += .01/CANVAS_SCALE*strafe_field_strength;

    vec3 sp = vec3(e.px,e.py,e.pz);
    vec3 n = scene(sp).x*-.01*sdf_normal(sp);
    //e.px+=n.x;
    //e.py+=n.y;
    //e.pz+=n.z;
    //BOUNDARY_CONDITIONS_MODE:  0-1-2 == BOUNCE-RESET-WRAP
    float ca = canvas_resolution.x / canvas_resolution.y;
    float x_edge = sqrt(ca);
    float y_edge = 1.0 / sqrt(ca);
    float z_edge = 1.0; // Z always spans [-1, 1] for now
    int boundary_mode = get_particle_boundary_conditions();
    if(boundary_mode==0){
        //reflect particles off canvas boundaries
        if (e.px < -x_edge || e.px > x_edge){
            e.vx=-e.vx;
            e.px=edgeflect(e.px/x_edge)*x_edge;
        }
        if (e.py < -y_edge || e.py > y_edge){
            e.vy=-e.vy;
            e.py=edgeflect(e.py/y_edge)*y_edge;
        }
        if (!TESTING_MODE && canvas_3d_size.z > 1) {
            if (e.pz < -z_edge || e.pz > z_edge){
                e.vz=-e.vz;
                e.pz=edgeflect(e.pz/z_edge)*z_edge;
            }
        }
    }
    else if(boundary_mode==1){
        //reset to initial conditions
        bool out_xy = e.px<-x_edge||e.px>x_edge||e.py<-y_edge||e.py>y_edge;
        bool out_z = !TESTING_MODE && canvas_3d_size.z > 1 && (e.pz < -z_edge || e.pz > z_edge);
        if(out_xy || out_z){
            reset(index);
            return;//reset expects to be the last thing we do. It handles entity buffer storage
        }
    }
    else if(boundary_mode==2){
        //wrap: X wraps [-x_edge,x_edge], Y wraps [-y_edge, y_edge]
        e.px = x_edge * 2.0 * (fract(e.px / (x_edge * 2.0) - 0.5) - 0.5);
        e.py = y_edge * 2.0 * (fract(e.py / (y_edge * 2.0) - 0.5) - 0.5);
        if (!TESTING_MODE && canvas_3d_size.z > 1) {
            e.pz = z_edge * 2.0 * (fract(e.pz / (z_edge * 2.0) - 0.5) - 0.5);
        }
    }

    //Atomic splat to canvas: scale by (1-p)/p so canvas_update's *p gives net (1-p)*splat
    float trail_p = calculate_setting(TRAIL_PERSISTENCE_SETTING, vec2(e.px, e.py), cohort);
    trail_p = clamp(trail_p, 0.001, 0.999);
    float splat_scale = (1.0 - trail_p) / trail_p;

    ivec3 img_res_3d = imageSize(can_img);
    vec2 half_ext = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv_xy = vec2(e.px, e.py) / (2.0 * half_ext) + 0.5;
    float uv_z = e.pz * 0.5 + 0.5;

#ifdef LINEAR_MC_DEPOSIT
    // Continuous voxel position (subtract 0.5 so integer coords = voxel centers)
    vec3 voxel_f;
    voxel_f.xy = uv_xy * vec2(img_res_3d.xy) - 0.5;
    voxel_f.z = (img_res_3d.z <= 1) ? 0.0 : uv_z * float(img_res_3d.z) - 0.5;

    ivec3 base = ivec3(floor(voxel_f));
    vec3 frac = voxel_f - vec3(base);

    // Trilinear weights per axis
    float wx0 = 1.0 - frac.x, wx1 = frac.x;
    float wy0 = 1.0 - frac.y, wy1 = frac.y;
    float wz0 = 1.0 - frac.z, wz1 = frac.z;
    if (img_res_3d.z <= 1) { wz0 = 1.0; wz1 = 0.0; } // depth-1: all weight to z=0

    // Stochastic voxel selection: pick one corner of the 2x2x2 neighborhood
    // proportional to trilinear weights (Monte Carlo linear filter)
    float r = hash(vec2(float(gl_GlobalInvocationID.x), float(frame_count)));
    float cumulative = 0.0;
    ivec3 offset = ivec3(0);

    cumulative += wx0 * wy0 * wz0;
    if (r >= cumulative) {
        cumulative += wx1 * wy0 * wz0;
        if (r >= cumulative) {
            cumulative += wx0 * wy1 * wz0;
            if (r >= cumulative) {
                cumulative += wx1 * wy1 * wz0;
                if (r >= cumulative) {
                    cumulative += wx0 * wy0 * wz1;
                    if (r >= cumulative) {
                        cumulative += wx1 * wy0 * wz1;
                        if (r >= cumulative) {
                            cumulative += wx0 * wy1 * wz1;
                            if (r >= cumulative) {
                                offset = ivec3(1,1,1);
                            } else { offset = ivec3(0,1,1); }
                        } else { offset = ivec3(1,0,1); }
                    } else { offset = ivec3(0,0,1); }
                } else { offset = ivec3(1,1,0); }
            } else { offset = ivec3(0,1,0); }
        } else { offset = ivec3(1,0,0); }
    }

    ivec3 voxel = base + offset;
#else
    // Nearest-neighbor deposit (original behavior)
    int voxel_z = (img_res_3d.z <= 1) ? 0 : int(uv_z * float(img_res_3d.z));
    ivec3 voxel = ivec3(ivec2(uv_xy * vec2(img_res_3d.xy)), voxel_z);
#endif

    if (voxel.x >= 0 && voxel.x < img_res_3d.x &&
        voxel.y >= 0 && voxel.y < img_res_3d.y &&
        voxel.z >= 0 && voxel.z < img_res_3d.z) {
        imageAtomicAdd(can_img, voxel, f16vec4(splat_scale * e.vx, splat_scale * e.vy, splat_scale * e.vz, 0.0));
    }

    //Commit new entity state to buffers
    entities[index]=e;


}
