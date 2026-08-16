// Shared Fluoddity particle physics.
//
// Extracted verbatim from entity_update.glsl so the streamline "read-only
// particle" tracer evaluates exactly the same behaviour. Both shaders splice
// this in via shader_prepend(), the same way fourier4_4.glsl is included, so
// there is one definition of the physics rather than two that drift.
//
// Requires, from the including shader (prepended before this file):
//   - fourier4_4.glsl (FourierCenter, fourier_noise, hash)
// Provides: the Entity/Rule/PhysicsSetting structs, every uniform the physics
// reads, the get_particle_* accessors, calculate_setting(), get_can(),
// get_field(), and calculate_entity_behavior().
//
// The tracer declares no Entity buffer of its own; it reads rules[] read-only
// and never writes entities[].
//SAME STRUCT USED IN BRUSH.VERT AND CAM_BRUSH.VERT
struct Entity {
    vec2 pos;
    vec2 vel;
    float hue;
    float size;
    float padding[2];  // Align to 16-byte boundary
};  // Total: 32 bytes (8 floats)
struct Rule {
    FourierCenter centers[10];
};
// The streamline tracer is a read-only consumer of this physics: it never
// touches the entity buffer, so it defines STREAMLINE_READONLY to skip the
// declaration entirely rather than binding a buffer it must not write.
#ifndef STREAMLINE_READONLY
layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};
#endif
// Not marked readonly: entity_update writes rules[] back when WRITE_RULES is
// set for rule readback. The tracer simply never writes it.
layout(std430, binding = 2) buffer RuleBuffer {
    Rule rules[];
};
// SYNCHRONIZED: This struct must match canvas.frag
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
uniform float WORLD_SIZE;
uniform int frame_count;
uniform Rule target_rule;
uniform sampler2D canvas; //trails canvas
uniform sampler2D field_texture; // Force/Strafe field (.xy=force, .zw=strafe)
uniform bool advanced_drawing_resources_initialized; // True when field_texture has valid data
uniform float force_field_strength; // Multiplier for force field effects
uniform float strafe_field_strength; // Multiplier for strafe field effects
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
uniform float HUE_SENSITIVITY;
uniform bool COLOR_BY_COHORT;
uniform bool DISABLE_SYMMETRY;
uniform int ABSOLUTE_ORIENTATION; // 0=Off, 1=Y axis, 2=Radial
uniform float ORIENTATION_MIX; // Blend factor for orientation calculations
uniform int BOUNDARY_CONDITIONS_MODE; //0-1-2 == BOUNCE-RESET-WRAP
uniform int RESET_MODE; //0-1-2 == GRID-RANDOM-RING
uniform int COHORTS; //each cohort gets its own rule and starting location
uniform float RULE_SEED;
uniform bool WRITE_RULES; // Set true for one frame when rule buffer readback is needed

// Multi-load control uniforms (small, stay as uniforms)
uniform int MULTILOAD_COUNT; // Number of loaded configs (0 = normal mode)
uniform float MULTI_LOAD_CURRENT_PROGRESS; // Current position in config ring (0-1)
uniform float MULTI_LOAD_SIMULTANEOUS_CONFIGS; // How many configs to span
uniform int MULTI_LOAD_ASSIGNMENT_MODE; // 0 = Cohorts, 1 = Random
uniform bool MULTI_LOAD_PER_CONFIG_INITIAL_CONDITIONS; // If true, use per-config reset modes
uniform bool MULTI_LOAD_PER_CONFIG_COHORTS; // If true, use per-config cohort counts
uniform bool MULTI_LOAD_PER_CONFIG_HAZARD_RATE; // If true, use per-config hazard rates

// Multi-load config data (large arrays, packed into SSBO)
struct MultiLoadConfig {
    // Physics parameters as PhysicsSetting structs (10 params * 6 floats = 60 floats)
    PhysicsSetting axial_force;
    PhysicsSetting lateral_force;
    PhysicsSetting sensor_gain;
    PhysicsSetting mutation_scale;
    PhysicsSetting drag;
    PhysicsSetting strafe_power;
    PhysicsSetting sensor_angle;
    PhysicsSetting global_force_mult;
    PhysicsSetting sensor_distance;
    PhysicsSetting hazard_rate;

    // Simulation settings (6 ints)
    int disable_symmetry;      // bool as int for alignment
    int absolute_orientation;  // 0=Off, 1=Y axis, 2=Radial
    int boundary_conditions;
    int reset_mode;
    int cohorts;
    int color_by_cohort;       // bool as int for alignment

    // Appearance and orientation mix (3 floats)
    float hue_sensitivity;
    float orientation_mix;
    float rule_seed;
};

layout(std430, binding = 3) buffer MultiLoadConfigBuffer {
    MultiLoadConfig configs[64];
};

// Multi-load target rules (separate buffer for cleaner organization)
layout(std430, binding = 4) buffer MultiLoadRuleBuffer {
    Rule target_rules[64];
};

////////////////////////////CONSTANTS
#define PI 3.1415926
#define ACTIVE_COUNT (600000*WORLD_SIZE) //Supports up to the size of the entity buffer.
#define SQRT_WORLD_SIZE (sqrt(WORLD_SIZE))
// Multi-load helper: Calculate which config index this particle should use
int get_particle_config_index() {
    if (MULTILOAD_COUNT == 0) return -1; // Not in multi-load mode

    // Calculate normalized index (0 to 1) for this particle
    float normalized_index = float(gl_GlobalInvocationID.x) / float(ACTIVE_COUNT);

    // For "Random" assignment mode, hash the normalized_index for stable pseudo-random assignment
    if (MULTI_LOAD_ASSIGNMENT_MODE == 1) {
        normalized_index = hash(vec2(normalized_index, 0.0));
    }

    // Calculate config index using circular ring formula
    // If SIMULTANEOUS_CONFIGS == 2, span across 2 full indices as normalized_index sweeps 0 to 1
    float offset = MULTI_LOAD_SIMULTANEOUS_CONFIGS / float(MULTILOAD_COUNT) * normalized_index;
    float ring_position = fract(MULTI_LOAD_CURRENT_PROGRESS + offset);
    int config_index = int(floor(float(MULTILOAD_COUNT) * ring_position));

    // Clamp to valid range
    return clamp(config_index, 0, MULTILOAD_COUNT - 1);
}
                            //Entities with index > ACTIVE_COUNT aren't rendered or updated
int get_particle_cohorts() {
    int idx = get_particle_config_index();
    // Use per-config value only if multi-load is active AND per-config checkbox is enabled
    if (idx >= 0 && MULTI_LOAD_PER_CONFIG_COHORTS) {
        return configs[idx].cohorts;
    }
    return COHORTS;
}
//Calculate the actual setting value for this particle. When sweeps are
//active, physics settings can depend on entity position and cohort
// SYNCHRONIZED: This function must match canvas.frag and sim.py::calculate_setting
// Locations to synchronize: shaders/entity_update.glsl, shaders/canvas.frag, sim.py
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



// Helper functions to get config values (return array value if multi-load, else single uniform)

PhysicsSetting get_particle_axial_force() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].axial_force : AXIAL_FORCE_SETTING;
}

PhysicsSetting get_particle_lateral_force() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].lateral_force : LATERAL_FORCE_SETTING;
}

PhysicsSetting get_particle_sensor_gain() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_gain : SENSOR_GAIN_SETTING;
}

PhysicsSetting get_particle_mutation_scale() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].mutation_scale : MUTATION_SCALE_SETTING;
}

PhysicsSetting get_particle_drag() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].drag : DRAG_SETTING;
}

PhysicsSetting get_particle_strafe_power() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].strafe_power : STRAFE_POWER_SETTING;
}

PhysicsSetting get_particle_sensor_angle() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_angle : SENSOR_ANGLE_SETTING;
}

PhysicsSetting get_particle_global_force_mult() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].global_force_mult : GLOBAL_FORCE_MULT_SETTING;
}

PhysicsSetting get_particle_sensor_distance() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].sensor_distance : SENSOR_DISTANCE_SETTING;
}

bool get_particle_disable_symmetry() {
    int idx = get_particle_config_index();
    return idx >= 0 ? bool(configs[idx].disable_symmetry) : DISABLE_SYMMETRY;
}

int get_particle_absolute_orientation() {
    int idx = get_particle_config_index();
    return idx >= 0 ? int(configs[idx].absolute_orientation) : ABSOLUTE_ORIENTATION;
}
PhysicsSetting get_particle_hazard_rate() {
    int idx = get_particle_config_index();
    return idx >= 0 &&MULTI_LOAD_PER_CONFIG_HAZARD_RATE? (configs[idx].hazard_rate) : HAZARD_RATE_SETTING;
}

//HARDCODED TO BE GLOBAL FOR NOW
int get_particle_boundary_conditions() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? configs[idx].boundary_conditions : BOUNDARY_CONDITIONS_MODE;
    return BOUNDARY_CONDITIONS_MODE;
}

int get_particle_reset_mode() {
    int idx = get_particle_config_index();
    // Use per-config value only if multi-load is active AND per-config checkbox is enabled
    if (idx >= 0 && MULTI_LOAD_PER_CONFIG_INITIAL_CONDITIONS) {
        return configs[idx].reset_mode;
    }
    return RESET_MODE;
}


//HARDCODED TO BE GLOBAL FOR NOW
float get_particle_hue_sensitivity() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? configs[idx].hue_sensitivity : HUE_SENSITIVITY;
    return HUE_SENSITIVITY;
}

//HARDCODED TO BE GLOBAL FOR NOW
bool get_particle_color_by_cohort() {
    //int idx = get_particle_config_index();
    //return idx >= 0 ? bool(configs[idx].color_by_cohort) : COLOR_BY_COHORT;
    return COLOR_BY_COHORT;
}

float get_particle_rule_seed() {
    int idx = get_particle_config_index();
    return idx >= 0 ? configs[idx].rule_seed : RULE_SEED;
}

Rule get_particle_target_rule() {
    int idx = get_particle_config_index();
    return idx >= 0 ? target_rules[idx] : target_rule;
}


////////////////////////////////////
//FOURIER NOISE IS IMPORTED INTO THIS SHADER
//FROM fourier4_4.glsl
////////////////////////////////////

//rotate p around origin by angle a
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}


//convert p (entity space) to texture coords and retrieve canvas (RG32F: velocity only)
vec2 canvas_uv(vec2 p){
    vec2 res=textureSize(canvas,0);
    float ca = res.x / res.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = p / (2.0 * half_extent) + 0.5;
    if(get_particle_boundary_conditions() == 2) uv = fract(uv);
    return uv;
}
vec2 get_can(vec2 p){
    return texture(canvas, canvas_uv(p)).rg;
}
#ifdef STREAMLINE_READONLY
// The tracer runs many steps between canvas updates, so it samples a blend of
// the previous and current frame instead of a piecewise-constant field. See
// the note in streamline_trace.glsl.
uniform sampler2D canvas_prev;
uniform bool FIELD_INTERPOLATE;
float g_field_alpha = 1.0;   // Set per step by the tracer's main loop.
vec2 get_can_lerp(vec2 p){
    vec2 uv = canvas_uv(p);
    vec2 cur = texture(canvas, uv).rg;
    if(!FIELD_INTERPOLATE) return cur;
    return mix(texture(canvas_prev, uv).rg, cur, clamp(g_field_alpha, 0.0, 1.0));
}
#endif
vec4 get_field(vec2 p){
    if(!advanced_drawing_resources_initialized)return vec4(0);
    vec2 res=textureSize(field_texture,0);
    float ca = res.x / res.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = p / (2.0 * half_extent) + 0.5;
    if(get_particle_boundary_conditions() == 2) uv = fract(uv);
    return texture(field_texture, uv);
}

vec2 safenorm(vec2 p){
    return length(p)==0?vec2(0):normalize(p);
}

float get_cohort(uint index) {
    return float(get_particle_cohorts()) * float(index) / float(ACTIVE_COUNT);
}

//Return all entities to their initialization state
void reset(uint index){

    float size=index<ACTIVE_COUNT?.0015/SQRT_WORLD_SIZE: 0;
    float cohort_val = get_cohort(index);
    float aspect = sqrt(canvas_resolution.x/canvas_resolution.y);

    //set pos and vel to random values on a small disk
    float cohort_scale = 0.019;//Size of each disk
    vec2 pos=cohort_scale*vec2(hash(vec2(cohort_val)),hash(vec2(cohort_val+index+2.142)));
    vec2 vel=.00005*(vec2(hash(vec2(cohort_val,index)),hash(vec2(cohort_val,pos.y)))*2-1);

    //RESET_MODE: 0=Grid, 1=Random, 2=Ring
    int reset_mode = get_particle_reset_mode();
    int cohorts = get_particle_cohorts();
    if(reset_mode == 0) {
        //GRID: position different cohorts at different places in a grid
        float spots=float(cohorts);
        float spot_rows=ceil(aspect*sqrt(spots));
        vec2 gridcell=vec2(int(cohort_val)%int(spot_rows),(int(cohort_val))/int(spot_rows));
        //pR(pos,floor(cohort_val)*3.1415*2*spots);
        //this aspect transform is good enough, but not perfect
        pos+=1.8*((gridcell)/spot_rows)*vec2(aspect);
        pos+= 1.8*(1/2.*(1./vec2(spot_rows,spots/spot_rows)-1))*vec2(aspect,1/aspect);
    }
    else if(reset_mode == 1) {
        //RANDOM: scatter cohorts randomly across the canvas, homogenous start
        pos= vec2(hash(vec2(cohort_val, 1.0)), hash(vec2(cohort_val, 2.0))) * 2.0 - 1.0;
        pos.x*=aspect;
        pos.y/=aspect;
    }
    else if(reset_mode == 2) {
        //RING: arrange cohorts in a ring pattern
        float angle = cohort_val / float(cohorts) * 2.0 * PI;
        float radius = 0.5;
        pos += vec2(cos(angle), sin(angle)) * radius;
        //pos += 0.02 * vec2(hash(vec2(cohort_val)), hash(vec2(cohort_val + 1.0))); // Small jitter
    }

    
    //store to persistent entity buffer
#ifndef STREAMLINE_READONLY
    entities[index]=Entity(pos,vel, 0.50, size, float[2](0,0));
#endif
}

//randomly change noise function parameters, scaled by parameter amount. 
//Each cohort gets a unique mutation for any given rule
void mutate_rule(inout Rule current_rule,float amount,float cohort){
    float seed = hash(current_rule.centers[4].frequency.xy+current_rule.centers[7].amplitude.yx+current_rule.centers[1].frequency.zw)+cohort;

    for(int i = 0; i < 10; i++) {
        vec4 amp_mutation = amount * (-1.0 + 2.0 * hash4(-.5+vec2(-i+seed,i)));
        current_rule.centers[i].amplitude += amp_mutation;
        current_rule.centers[i].frequency *= 1 + amount * 0.5 * (hash(vec2(seed,i))-.5);
    }
}


//Used to enforce left-right symmetry in the local coordinates vec2(forward, left)
vec2 y_reflect(vec2 p){
    return p*vec2(1,-1);
}
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
    return (fourier_noise(rule.centers, vec4(L,R)));
}



//This function determines entity output by plugging sensor values into a noise function called black_box()
//The calculation is performed twice, once in mirrored coordinates, and the two values are averaged.
//This keeps entities from displaying clockwise/counterclockwise bias.
//PARAMETERS:
//--L and R: velocity field measurements from left sensor and right sensor.
//--axis: forward vector that defines our orientation.
//--rule: coefficients for the noise function that dictates entity behavior.
//--pos: entity position (for parameter sweeps)
//--cohort: entity cohort (for parameter sweeps)
//RETURNS:
//--force: A "push" vector that will be added to entity.vel
//--strafe: A "hop" vector that will be added to entity.pos and have no effect on velocity
//--color: vec2 to be used as parameters in a coloring function
void calculate_entity_behavior( vec2 L,vec2 R, vec2 axis, Rule rule, vec2 pos, float cohort, out vec2 force, out vec2 strafe, out vec2 color){

    //build a local coordinate frame where "axis" is forward.
    vec2 forward=safenorm(axis);
    vec2 left=vec2(forward.y,-forward.x);

    //Convert L and R to local coordinates.
    //Ie. decompose each into an axial component and a lateral component
    L=vec2(dot(L,forward),dot(L,left));
    R=vec2(dot(R,forward),dot(R,left));

    //calculate black box noise values
    vec4 baseterm= black_box(L,R,rule);
    vec4 mirrorterm=black_box(y_reflect(R),y_reflect(L),rule);
    if(DISABLE_SYMMETRY){mirrorterm = vec4(0);}//disable symmetry by zeroing the mirror term

    //Combine base and mirror terms
    force = baseterm.xy+y_reflect(mirrorterm.xy);
    strafe = baseterm.zw + y_reflect(mirrorterm.zw);

    //Convert force and strafe back to world coordinates
    force=forward*force.x*calculate_setting(get_particle_axial_force(),pos,cohort)+left*force.y*calculate_setting(get_particle_lateral_force(),pos,cohort);
    strafe = forward*strafe.x*calculate_setting(get_particle_axial_force(),pos,cohort) + left * strafe.y * calculate_setting(get_particle_lateral_force(),pos,cohort);

    color = baseterm.xy+(mirrorterm.xy); //Just an arbitrary function of blackbox output. Reuses force terms.
    return;
}
