#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// Visualize the lottery canvas: color each pixel by its winning entity's hue.
// Only dispatched when the "Lottery Canvas" view is selected.

#define INDEX_BITS 20
#define INDEX_MASK ((1u << INDEX_BITS) - 1u)

// MUST match the 32-byte Entity struct in entity_update.glsl.
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

layout(r32ui, binding = 0) readonly uniform uimage2D lotto_canvas;
layout(rgba32f, binding = 1) writeonly uniform image2D display_tex;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(lotto_canvas);
    if (pixel.x >= size.x || pixel.y >= size.y) return;

    uint ticket = imageLoad(lotto_canvas, pixel).r;
    if (ticket == 0u) {
        imageStore(display_tex, pixel, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    uint winner_index = ticket & INDEX_MASK;
    float hue = entities[winner_index].hue;
    vec3 rgb = hsv2rgb(vec3(hue, 0.8, 1.0));
    imageStore(display_tex, pixel, vec4(rgb, 1.0));
}
