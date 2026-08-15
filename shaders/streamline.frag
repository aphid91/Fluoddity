#version 430

uniform vec3 line_color;
uniform float line_opacity;

in float v_t;        // 0 at the oldest visible point, 1 at the newest
flat in int v_dead;  // Vertices with no history behind them yet
out vec4 fragColor;

void main() {
    // Surplus vertices on a particle that has not filled its tail yet.
    if (v_dead != 0) {
        discard;
    }

    // Brighten the leading end so direction of travel reads at a glance, and
    // fade the tail out so overwritten history disappears gracefully.
    float head = mix(0.35, 1.35, v_t);
    fragColor = vec4(line_color * head, line_opacity * mix(0.25, 1.0, v_t));
}
