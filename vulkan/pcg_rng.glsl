#ifndef PCG_RNG_GLSL
#define PCG_RNG_GLSL

#include "util.glsl"

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

uint32_t pcg32_random(inout PCGRand rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005UL + 1;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = uint32_t(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float pcg32_randomf(inout PCGRand rng) {
	return ldexp(float(pcg32_random(rng)), -32);
}

PCGRand get_rng(int frame_id) {
	uvec2 pixel = uvec2(gl_LaunchIDNV.xy);
	uint32_t seed = (pixel.x + pixel.y * gl_LaunchSizeNV.x) * (frame_id + 1);

	PCGRand rng;
	rng.state = 0;
	pcg32_random(rng);
	rng.state += seed;
	pcg32_random(rng);
	return rng;
}

#endif

