#version 450
layout(local_size_x = 64) in;

// Lottery payout: selection + reproduction pass. Runs once per entity, after
// entity_update.glsl has filled lotto_canvas with per-pixel winning tickets.
//   Pixel WINNER  -> mutates its own rule slightly (drift).
//   Pixel LOSER   -> adopts the winner's rule (successful rules spread).
// Cohort/color adoption is intentionally dropped (see plan): cohort is now
// index-derived and color is a single hue field. Rule-only adoption.

#define INDEX_BITS 20
#define INDEX_MASK ((1u << INDEX_BITS) - 1u)

// fourier4_4.glsl (prepended) provides: FourierCenter, Rule, hash, hash4, pcg_hash.

// MUST match the 32-byte Entity struct in entity_update.glsl / brush.vert / cam_brush.vert.
struct Entity {
    vec2 pos;
    vec2 vel;
    float hue;
    float size;
    float padding[2];
};  // 32 bytes

layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};

struct Rule {
    FourierCenter centers[10];
};
layout(std430, binding = 2) buffer RuleBuffer {
    Rule rules[];
};

layout(r32ui, binding = 0) readonly uniform uimage2D lotto_canvas;

uniform sampler2D canvas; //trails canvas (RG32F: velocity only), bound to texture unit 1
uniform vec2 canvas_resolution;
uniform float WORLD_SIZE;

//convert p (entity space) to texture coords and retrieve canvas (RG32F: velocity only)
//NOTE: mirror of get_can() in entity_update.glsl (boundary-fract omitted; add if needed).
vec2 get_can(vec2 p){
    vec2 res=textureSize(canvas,0);
    float ca = res.x / res.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = p / (2.0 * half_extent) + 0.5;
    return texture(canvas, uv).rg;
}
#define ACTIVE_COUNT (600000*WORLD_SIZE)

// Generic scratch uniforms for live-coding (Generics window sliders, -1..1).
uniform vec4 generic03;
uniform vec4 generic47;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ENTITY_COUNT || index >= uint(ACTIVE_COUNT)) return;

    Entity e = entities[index];

    // Worldspace -> pixel. Identical transform to entity_update.glsl's ticket write:
    //   x_edge = sqrt(ca), y_edge = 1/sqrt(ca)
    float ca = canvas_resolution.x / canvas_resolution.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    vec2 uv = e.pos / (2.0 * half_extent) + 0.5;
    ivec2 pixel = clamp(ivec2(uv * canvas_resolution), ivec2(0), ivec2(canvas_resolution) - 1);

    uint ticket = imageLoad(lotto_canvas, pixel).r;
    uint winner_index = ticket & INDEX_MASK;
if(hash(e.pos)<.7){return;}///////////////////////NO LOTTO FOR ME
    if (ticket != 0u && winner_index != index) {
        // Loser: adopt the winner's rule (whole-rule copy).
        float dens = length(get_can(e.pos).xy);
        float CHILD_MUT = .002;//.005;//.01*abs(dens-generic03.x);
        Rule new_rule = rules[winner_index];
        int r0 = int(hash(e.pos)*10);
        int r2 = int(hash(10-e.pos)*10);
        vec4 r1 = CHILD_MUT * (hash4(e.vel)*2.-1.);//.001
        vec4 r3 = CHILD_MUT * (hash4(10-e.vel)*2.-1.);//.001
        new_rule.centers[r0].amplitude+=r1;
        new_rule.centers[r2].frequency+=r3;
        rules[index] = new_rule;//rules[winner_index];
        entities[index].hue = entities[winner_index].hue+(r1.x+r3.y)*.21;
    } else if (winner_index == index) {
        // Winner: nudge a couple of coefficients so successful lineages keep drifting.
        //int r0 = int(hash(e.pos) * 10.0);
        //int r2 = int(hash(10.0 - e.pos) * 10.0);
        //vec4 r1 = 0.031 * (hash4(e.vel) * 2.0 - 1.0);
        //vec4 r3 = 0.031 * (hash4(10.0 - e.vel) * 2.0 - 1.0);
        //rules[index].centers[r0].amplitude += r1;
        //rules[index].centers[r2].frequency += r3;
    }
}
