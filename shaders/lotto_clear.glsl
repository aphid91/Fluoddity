#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// Zero the lottery canvas before entity_update writes new tickets.
layout(r32ui, binding = 0) uniform uimage2D lotto_canvas;

// Generic scratch uniforms for live-coding (Generics window sliders, -1..1).
uniform vec4 generic03;
uniform vec4 generic47;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(lotto_canvas);
    if (pixel.x >= size.x || pixel.y >= size.y) return;
    imageStore(lotto_canvas, pixel, uvec4(0));
}
