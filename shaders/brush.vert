#version 430 
        
uniform vec2 canvas_resolution;
//SYNC WITH ENTITY_UPDATE.GLSL AND CAM_BRUSH.VERT
struct Entity {
    vec2 pos;
    vec2 vel;
    float hue;
    float size;
    float padding[2];  // Align to 16-byte boundary
};  // Total: 32 bytes (8 floats)
layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};

out vec2 uv;
out vec4 pos_vel;
out vec4 view_col;
void main() {
    int instance_id = gl_InstanceID;
    int vertex_id = gl_VertexID;
    
    // Read entity position
    vec2 entity_pos = entities[instance_id].pos;
    vec2 entity_vel = entities[instance_id].vel;
    // Generate quad vertices based on vertex_id (0-3)
    // Create small square centered at entity position
    float size = entities[instance_id].size;
    vec2 offsets[4] = vec2[](
        vec2(-size, -size),  // bottom-left
        vec2( size, -size),  // bottom-right
        vec2( size,  size),  // top-right
        vec2(-size,  size)   // top-left
    );
    vec2 uv_coords[4] = vec2[](
        vec2(0,0),
        vec2(1,0),
        vec2(1,1),
        vec2(0,1)
    );
    vec2 particle_uv=uv_coords[vertex_id];
    vec2 vertex_pos = entity_pos + offsets[vertex_id];
    
    // Entity space to clip space: entity bounds [-x_edge,x_edge]x[-y_edge,y_edge] -> [-1,1]^2
    float ca = canvas_resolution.x / canvas_resolution.y;
    gl_Position = vec4(vertex_pos, 0.0, 1.0) * vec4(1.0/sqrt(ca), sqrt(ca), 1, 1);

    uv = particle_uv;
    pos_vel=vec4(entity_pos,entity_vel);
    view_col=vec4(entities[instance_id].hue, 0.8, 1.0, 0.045);
}