#version 430

// Audio mix reduction.
//
// Sums the per-voice lanes written by the tracer into a single mono block.
// One invocation per sample, walking the voice axis - no atomics, so the
// summation order is fixed and the mix is bit-reproducible run to run.
//
// Voices are near-independent in practice (measured |correlation| ~0.015
// mean, 0.058 max between particles), so their energy adds as sqrt(n) rather
// than n. Normalising by 1/sqrt(n) therefore keeps the perceived level
// roughly constant as Voice Count changes; 1/n would make the mix quieter
// with every voice added.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 7) readonly buffer StreamlineAudio {
    float audio_samples[];
};

layout(std430, binding = 8) writeonly buffer StreamlineAudioMix {
    float mix_samples[];
};

uniform int VOICE_COUNT;      // Lanes to sum
uniform int LANE_STRIDE;      // Floats per voice lane
uniform int SLOT_BASE;        // Start of this block within a lane
uniform int BLOCK_SIZE;       // Samples in this block
uniform float MIX_GAIN;       // Usually 1/sqrt(VOICE_COUNT)

void main() {
    int k = int(gl_GlobalInvocationID.x);
    if (k >= BLOCK_SIZE) {
        return;
    }

    float acc = 0.0;
    for (int v = 0; v < VOICE_COUNT; ++v) {
        acc += audio_samples[v * LANE_STRIDE + SLOT_BASE + k];
    }

    mix_samples[SLOT_BASE + k] = acc * MIX_GAIN;
}
