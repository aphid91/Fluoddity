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

// Per-voice delay history, written by the tracer. Read instead of the lanes
// whenever a voice is delayed: the lane slots are recycled as soon as the
// readback frees them, so they hold only the block or two still in flight.
layout(std430, binding = 9) readonly buffer StreamlineAudioHistory {
    float audio_history[];
};

uniform int VOICE_COUNT;      // Lanes to sum
uniform int LANE_STRIDE;      // Floats per voice lane
uniform int SLOT_BASE;        // Start of this block within a lane
uniform int BLOCK_SIZE;       // Samples in this block
uniform float MIX_GAIN;       // Usually 1/sqrt(VOICE_COUNT)

// --- Per-voice delay ---
// Each voice is read a fixed distance in the past, so the voices smear
// against each other instead of landing together. The distance is hashed
// from the voice index rather than stored: a table would need an SSBO to
// reach 8192 voices, and the hash is just as deterministic.
uniform bool DELAY_ENABLED;
uniform int DELAY_MIN;        // Samples, inclusive
uniform int DELAY_MAX;        // Samples, inclusive. Clamped by the caller to
                              // MAX_VOICE_DELAY_SAMPLES.
uniform int HISTORY_LEN;      // Samples per voice in the history ring
uniform int HISTORY_CURSOR;   // Total samples written before this block

uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Delay for voice v, in samples. Stable for the life of the stream: it
// depends only on the voice index, so a voice does not change pitch by
// having its delay drift under it.
int voice_delay(int v) {
    if (!DELAY_ENABLED || DELAY_MAX <= 0) {
        return 0;
    }
    int span = DELAY_MAX - DELAY_MIN;
    if (span <= 0) {
        return DELAY_MIN;
    }
    return DELAY_MIN + int(hash_u32(uint(v) * 0x9e3779b9u) % uint(span + 1));
}

void main() {
    int k = int(gl_GlobalInvocationID.x);
    if (k >= BLOCK_SIZE) {
        return;
    }

    float acc = 0.0;

    if (!DELAY_ENABLED) {
        // Undelayed: read the lane the tracer just wrote. Keeps the common
        // case a contiguous read with no history indirection.
        for (int v = 0; v < VOICE_COUNT; ++v) {
            acc += audio_samples[v * LANE_STRIDE + SLOT_BASE + k];
        }
    } else {
        // Delayed: read the history ring, which is never reclaimed. The
        // cursor is where this block STARTS in history, so sample k of it
        // sits at cursor + k, and the delay steps back from there.
        for (int v = 0; v < VOICE_COUNT; ++v) {
            int rd = (HISTORY_CURSOR + k - voice_delay(v)) & (HISTORY_LEN - 1);
            acc += audio_history[v * HISTORY_LEN + rd];
        }
    }

    mix_samples[SLOT_BASE + k] = acc * MIX_GAIN;
}
