#pragma once

#include "cuda_utils.h"

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

__device__ uint32_t pcg32_random(PCGRand &rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005ULL + 1;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ float pcg32_randomf(PCGRand &rng) {
	return ldexp((float)pcg32_random(rng), -32);
}

__device__ PCGRand get_rng(uint32_t seed) {
	PCGRand rng;
	rng.state = 0;
	pcg32_random(rng);
	rng.state += seed;
	pcg32_random(rng);
	return rng;
}

