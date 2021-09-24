#pragma once

#include <math_constants.h>
#include <optix.h>
#include "float3.h"
#include "types.h"

__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

__device__ float luminance(const float3 &c) {
	return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ float pow2(float x) {
	return x * x;
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

template<typename T>
__device__ T clamp(const T &x, const T &lo, const T &hi) {
	if (x < lo) {
		return lo;
	}
	if (x > hi) {
		return hi;
	}
	return x;
}

__device__ float lerp(float x, float y, float s) {
	return x * (1.f - s) + y * s;
}

__device__ float3 lerp(float3 x, float3 y, float s) {
	return x * (1.f - s) + y * s;
}

__device__ float3 reflect(const float3 &i, const float3 &n) {
	return i - 2.f * n * dot(i, n);
}

__device__ float3 refract_ray(const float3 &i, const float3 &n, float eta) {
	float n_dot_i = dot(n, i);
	float k = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
	if (k < 0.f) {
		return make_float3(0.f);
	}
	return eta * i - (eta * n_dot_i + sqrt(k)) * n;
}

__device__ float component(const float4 &v, const uint32_t i) {
    switch (i) {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
    case 3: return v.w;
    default: return CUDART_NAN_F;
    }
}

__device__ void* unpack_ptr(uint32_t hi, uint32_t lo) {
	const uint64_t val = static_cast<uint64_t>(hi) << 32 | lo;
	return reinterpret_cast<void*>(val);
}

__device__ void pack_ptr(void *ptr, uint32_t &hi, uint32_t &lo) {
	const uint64_t val = reinterpret_cast<uint64_t>(ptr);
	hi = val >> 32;
	lo = val & 0x00000000ffffffff;
}

template<typename T>
__device__ T& get_payload() {
	return *reinterpret_cast<T*>(unpack_ptr(optixGetPayload_0(), optixGetPayload_1()));
}

template<typename T>
__device__ const T& get_shader_params() {
	return *reinterpret_cast<const T*>(optixGetSbtDataPointer());
}
