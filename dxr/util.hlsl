#ifndef UTIL_HLSL
#define UTIL_HLSL

#include "kernels/wrappers.h"

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define NUM_RAY_TYPES 2
#define MAX_PATH_DEPTH 5

struct HitInfo {
	float4 color_dist;
	float4 normal;
};

struct OcclusionHitInfo {
	int hit;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
	float2 bary;
};

#endif

