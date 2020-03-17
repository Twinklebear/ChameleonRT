#pragma once

#include "cuda_utils.h"

// https://github.com/ospray/ospray/blob/master/ospray/math/random.ih
struct LCGRand {
    uint32_t state;
};

__device__ uint32_t murmur_hash3_mix(uint32_t hash, uint32_t k)
{
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint32_t r1 = 15;
    const uint32_t r2 = 13;
    const uint32_t m = 5;
    const uint32_t n = 0xe6546b64;

    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;

    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;

    return hash;
}

__device__ uint32_t murmur_hash3_finalize(uint32_t hash)
{
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;

    return hash;
}

__device__ uint32_t lcg_random(LCGRand &rng)
{
    const uint32_t m = 1664525;
    const uint32_t n = 1013904223;
    rng.state = rng.state * m + n;
    return rng.state;
}

__device__ float lcg_randomf(LCGRand &rng)
{
    return ldexp((float)lcg_random(rng), -32);
}

__device__ LCGRand get_rng(int frame_id)
{
    const uint2 pixel = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint2 dims = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);

    LCGRand rng;
    rng.state = murmur_hash3_mix(0, pixel.x + pixel.y * dims.x);
    rng.state = murmur_hash3_mix(rng.state, frame_id);
    rng.state = murmur_hash3_finalize(rng.state);

    return rng;
}

