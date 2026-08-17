#version 430

// Time-smoothed canvas ("audio-driver field").
//
// out = mix(previous_smoothed, real_canvas, SMOOTH_AMOUNT)
//
// A per-physics-step exponential moving average of the canvas. The canvas
// carries a lot of temporal noise: streamers fall into a structure and start
// to resonate, but the field churns underneath them so the tone never
// settles. Running the tracer on a frozen canvas gives clean tones and no
// evolution; this is the middle ground - the structures persist long enough
// to ring, while the field still advances.
//
// Both channels are the canvas vector field (RG), so this smooths the vectors
// componentwise rather than a magnitude. That keeps direction and magnitude
// consistent: averaging magnitudes separately would let a smoothed direction
// disagree with a smoothed speed.

in vec2 texcoord;
out vec2 frag_color;

uniform sampler2D real_canvas;      // The live canvas, this step
uniform sampler2D prev_smoothed;    // This field's own previous output
uniform float SMOOTH_AMOUNT;        // 0 = frozen, 1 = follow the canvas exactly
uniform bool SEED_FROM_CANVAS;      // Ignore history; copy the canvas straight

void main() {
    vec2 cur = texture(real_canvas, texcoord).rg;

    if (SEED_FROM_CANVAS) {
        // First step after an allocate or a reset. Starting from zero instead
        // would make the field fade in from a dead canvas, and every streamer
        // would sit motionless until the average caught up.
        frag_color = cur;
        return;
    }

    vec2 prev = texture(prev_smoothed, texcoord).rg;
    frag_color = mix(prev, cur, clamp(SMOOTH_AMOUNT, 0.0, 1.0));
}
