#version 430

uniform vec3 line_color;
uniform float line_opacity;

in float v_t;        // 0 at the seed, 1 at the end of the path
flat in int v_dead;  // Vertices past the streamline's valid point count
out vec4 fragColor;

void main() {
    // Degenerate segments past the end of a short streamline.
    if (v_dead != 0) {
        discard;
    }

    // Brighten the head of the path so direction of travel reads at a glance.
    float head = mix(1.35, 0.55, v_t);
    fragColor = vec4(line_color * head, line_opacity);
}
