#version 430

in vec2 uv;
in vec4 pos_vel;
in vec4 view_col;
out vec2 brush_out;  // RG32F output: velocity only

uniform int frame_count;
uniform float trail_persistence;

float gaussian(vec2 pos, float sigma) {
    float sigma2 = sigma * sigma;
    float norm = 1.0 / (2.0 * 3.14159265359 * sigma2);
    float exponent = -(dot(pos, pos)) / (2.0 * sigma2);
    return norm * exp(exponent);
}

void main() {
    if (frame_count == 0) {
        brush_out = vec2(0);
        return;
    }

    float kernel_func = gaussian(uv - .5, .163);
    if (length(uv - .5) > .5 || view_col.w == 0) { discard; }
    vec2 vel = pos_vel.zw;

    // The old pipeline used SRC_ALPHA blending with alpha = kernel_func,
    // which effectively squared the kernel (once in output, once via alpha blend).
    // With ONE,ONE blending we must square it explicitly to match.
    // Scale by (1-p) to match old blend: canvas = blur(canvas)*p + (1-p)*brush
    float p = clamp(trail_persistence, 0.001, 0.999);
    float prescale = (1.0 - p);

    brush_out = vel * prescale * kernel_func * kernel_func;
}
