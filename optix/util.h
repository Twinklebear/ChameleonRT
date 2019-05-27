#pragma once

#include <math_constants.h>
#include <optix.h>
#include <optix_math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

#define EPSILON 0.0001

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define MAX_PATH_DEPTH 5

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

__device__ void ortho_basis(float3 &v_x, float3 &v_y, const float3 &n) {
	v_y = make_float3(0.f, 0.f, 0.f);

	if (n.x < 0.6f && n.x > -0.6f) {
		v_y.x = 1.f;
	} else if (n.y < 0.6f && n.y > -0.6f) {
		v_y.y = 1.f;
	} else if (n.z < 0.6f && n.z > -0.6f) {
		v_y.z = 1.f;
	} else {
		v_y.x = 1.f;
	}
	v_x = normalize(cross(v_y, n));
	v_y = normalize(cross(n, v_x));
}

__device__ float pow2(float x) {
	return x * x;
}

__device__ bool all_zero(const float3 &v) {
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

__device__ float3 refract_ray(const float3 &i, const float3 &n, float eta) {
	float n_dot_i = dot(n, i);
	float k = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
	if (k < 0.f) {
		return make_float3(0.f);
	}
	return eta * i - (eta * n_dot_i + sqrt(k)) * n;
}

