#ifndef KERNELS_WRAPPERS_H
#define KERNELS_WRAPPERS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif
#define EPSILON 0.0001

#if defined(TARGET_HLSL)

#define IN(T, X) in const T X
#define OUT(T, X) out T X
#define INOUT(T, X) inout T X
#define varying
#define uniform
#define __device__

float2 make_float2(float s) {
	return float2(s, s);
}

float2 make_float2(float x, float y) {
	return float2(x, y);
}

float3 make_float3(float s) {
	return float3(s, s, s);
}

float3 make_float3(float x, float y, float z) {
	return float3(x, y, z);
}

float3 neg(in const float3 v) { return -v; }

#elif defined(TARGET_CUDA)

#define IN(T, X) const T &X
#define OUT(T, X) T &X
#define INOUT(T, X) T &X
#define varying
#define uniform

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

__device__ float3 neg(const float3 &v) { return -v; }

#elif defined(TARGET_ISPC)

#define IN(T, X) const T &X
#define OUT(T, X) T &X
#define INOUT(T, X) T &X
#define __device__

typedef unsigned int8 uint8_t;
typedef unsigned int uint32_t;
typedef unsigned int64 uint64_t;

float saturate(float x) {
	return clamp(x, 0.f, 1.f);
}

float lerp(float x, float y, float s) {
	return x * (1.f - s) + y * s;
}

float3 lerp(const float3 &x, const float3 &y, float s) {
	return x * (1.f - s) + y * s;
}


#else
#error "No target specified for kernels!"
#endif


__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

__device__ void ortho_basis(OUT(float3, v_x), OUT(float3, v_y), IN(float3, n)) {
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

__device__ float luminance(IN(float3, c)) {
	return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ float pow2(float x) {
	return x * x;
}

__device__ bool all_zero(IN(float3, v)) {
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

// Utils from HLSL for ISPC and CUDA
__device__ float3 reflect_ray(IN(float3, i), IN(float3, n)) {
	return i - 2.f * n * dot(i, n);
}

__device__ float3 refract_ray(IN(float3, i), IN(float3, n), float eta) {
	float n_dot_i = dot(n, i);
	float k = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
	if (k < 0.f) {
		return make_float3(0.f);
	}
	return eta * i - (eta * n_dot_i + sqrt(k)) * n;
}

#endif

