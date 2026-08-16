#version 450
layout(local_size_x = 64) in;


void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ENTITY_COUNT) return;

    // Inactive entities get zeroed out. Position offscreen so they don't accidentally get clicked on
    if (index >= ACTIVE_COUNT) {
        entities[index] = Entity(vec2(10000), vec2(0), 0.0, 0.0, float[2](0,0));
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
    mutate_rule(current_rule,calculate_setting(get_particle_mutation_scale(),e.pos,cohort),get_particle_rule_seed()+floor(cohort));
    // Only write rules when explicitly requested (expensive - 320 bytes per particle)
    if(WRITE_RULES) {
        rules[index] = current_rule;
    }
    
    //frame_count == 0 signals a simulation reset
    if (frame_count==0||calculate_setting(get_particle_hazard_rate(),e.pos,cohort)>hash(vec2(float(index)/float(ACTIVE_COUNT),frame_count))){reset(index);return;}



    //Calculate position offsets for the two sensors.
    float sample_dist = 1./SQRT_WORLD_SIZE*.005 * calculate_setting(get_particle_sensor_distance(),e.pos,cohort);
    
    //variable sample distance?
    //sample_dist *= (get_can(e.pos).z*10);
    //GOOD 1./dot(normalize(e.vel),normalize(get_can(e.pos).xy));
    //length(e.vel)/.05;//length(get_can(e.pos).xy)/.01;
    
    int ORIENTATION_MODE =get_particle_absolute_orientation();
    float mix_amt = min(1,ORIENTATION_MODE)*ORIENTATION_MIX;
    vec2 orientation = safenorm(e.vel);//vector facing the same direction as velocity, with length==samplen
    if(ORIENTATION_MODE==1){orientation = mix(orientation,vec2(0,1),mix_amt);}
    else if(ORIENTATION_MODE==2){orientation = mix(orientation,-normalize(e.pos),mix_amt);}
    vec2 left_sensor_offset = orientation*sample_dist;
    vec2 right_sensor_offset = orientation*sample_dist;
    pR(left_sensor_offset,calculate_setting(get_particle_sensor_angle(),e.pos,cohort)*PI);//rotate them opposite directions
    pR(right_sensor_offset,-calculate_setting(get_particle_sensor_angle(),e.pos,cohort)*PI);

    //read the trails from canvas (RG32F: velocity only)
    vec2 ltap = get_can(e.pos+left_sensor_offset);
    vec2 rtap = get_can(e.pos+right_sensor_offset);

    
    //rescale sensor values
    float sensor_scaling = SQRT_WORLD_SIZE*38.855*calculate_setting(get_particle_sensor_gain(),e.pos,cohort);
    ltap *= sensor_scaling;
    rtap *= sensor_scaling;

    //compute entity action
    vec2 strafe =vec2(0);
    vec2 force = vec2(0);
    vec2 col_params = vec2(0);
    calculate_entity_behavior(ltap,rtap,orientation,current_rule,e.pos,cohort,force,strafe,col_params);

    //rescale output forces
    force *= 1./SQRT_WORLD_SIZE*calculate_setting(get_particle_global_force_mult(),e.pos,cohort)/400.;
    strafe *= 1./SQRT_WORLD_SIZE*calculate_setting(get_particle_global_force_mult(),e.pos,cohort)/20.;


    //Set entity hue. Saturation (.8), brightness (1.0) and alpha (0.045) were
    //always constant, so they are rebuilt in the vertex shaders instead of stored.
    e.hue = get_particle_hue_sensitivity()*col_params.x;//hue can be anything
    if(get_particle_color_by_cohort()) {e.hue = hash(vec2(floor(cohort)));} //just assign a random hue to each cohort

    //Accelerate: Apply drag and add force to e.vel,
    e.vel = e.vel*calculate_setting(get_particle_drag(),e.pos,cohort) + force;
    //Move: add e.vel and strafe to e.pos
    e.pos += e.vel;
    e.pos += strafe*calculate_setting(get_particle_strafe_power(),e.pos,cohort);

    //ADVANCED DRAWING force / strafe
    vec4 draw_sample =get_field(e.pos);
    e.vel += .01*force_field_strength*draw_sample.xy;
    e.pos += .01*strafe_field_strength*draw_sample.zw;

    //BOUNDARY_CONDITIONS_MODE:  0-1-2 == BOUNCE-RESET-WRAP
    float ca = canvas_resolution.x / canvas_resolution.y;
    float x_edge = sqrt(ca);
    float y_edge = 1.0 / sqrt(ca);
    int boundary_mode = get_particle_boundary_conditions();
    if(boundary_mode==0){
        //reflect particles off canvas boundaries
        if (e.pos.x < -x_edge || e.pos.x > x_edge){
            e.vel.x=-e.vel.x;
            e.pos.x=edgeflect(e.pos.x/x_edge)*x_edge;
        }
        if (e.pos.y < -y_edge || e.pos.y > y_edge){
            e.vel.y=-e.vel.y;
            e.pos.y=edgeflect(e.pos.y/y_edge)*y_edge;
        }
    }
    else if(boundary_mode==1){
        //reset to initial conditions
        if(e.pos.x<-x_edge||e.pos.x>x_edge||e.pos.y<-y_edge||e.pos.y>y_edge){
            reset(index);
            return;//reset expects to be the last thing we do. It handles entity buffer storage
        }
    }
    else if(boundary_mode==2){
        //wrap: X wraps [-x_edge,x_edge], Y wraps [-y_edge, y_edge]
        e.pos.x = x_edge * 2.0 * (fract(e.pos.x / (x_edge * 2.0) - 0.5) - 0.5);
        e.pos.y = y_edge * 2.0 * (fract(e.pos.y / (y_edge * 2.0) - 0.5) - 0.5);
    }

    //Commit new entity state to buffers
    entities[index]=e;


}
