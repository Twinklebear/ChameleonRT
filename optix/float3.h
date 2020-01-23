#pragma once

#include "types.h"

__device__ float4 make_float4(float c) {
	return make_float4(c, c, c, c);
}

__device__ float4 make_float4(float3 v, float c) {
	return make_float4(v.x, v.y, v.z, c);
}

__device__ float3 make_float3(float c) {
	return make_float3(c, c, c);
}

__device__ float3 make_float3(float4 v) {
	return make_float3(v.x, v.y, v.z);
}

__device__ float2 make_float2(float c) {
	return make_float2(c, c);
}

__device__ float2 make_float2(uint2 v) {
	return make_float2(v.x, v.y);
}

__device__ float length(const float3 &v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 normalize(const float3 &v) {
	float l = length(v);
	if (l < 0.f) {
		l = 0.0001f;
	}
	const float c = 1.f / length(v);
	return make_float3(v.x * c, v.y * c, v.z * c);
}

__device__ float3 cross(const float3 &a, const float3 &b) {
	float3 c;
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

__device__ float3 neg(const float3 &a) {
	return make_float3(-a.x, -a.y, -a.z);
}

__device__ bool all_zero(const float3 &v) {
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

__device__ float dot(const float3 a, const float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float4 operator*(const uint32_t s, const float4 &v) {
	return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

__device__ float4 operator*(const float4 &v, const uint32_t s) {
	return s * v;
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ float4 operator/(const float4 &a, const uint32_t s) {
	const float x = 1.f / s;
	return x * a;
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator-(const float3 &a, const float s) {
	return make_float3(a.x - s, a.y - s, a.z - s);
}

__device__ float3 operator-(const float s, const float3 &a) {
	return make_float3(s - a.x, s - a.y, s - a.z);
}

__device__ float3 operator-(const float3 &a) {
	return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator+(const float3 &a, const float s) {
	return make_float3(a.x + s, a.y + s, a.z + s);
}

__device__ float3 operator+(const float s, const float3 &a) {
	return a + s;
}

__device__ float3 operator*(const float3 &a, const float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 operator*(const float s, const float3 &a) {
	return a * s;
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 operator/(const float3 &a, const float s) {
	return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ float3 operator/(const float s, const float3 &a) {
	return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ float3 operator/(const float3 &a, const float3 &b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ float2 operator-(const float2 &a, const float2 &b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator-(const float2 &a, const float s) {
	return make_float2(a.x - s, a.y - s);
}

__device__ float2 operator-(const float s, const float2 &a) {
	return make_float2(s - a.x, s - a.y);
}

__device__ float2 operator-(const float2 &a) {
	return make_float2(-a.x, -a.y);
}

__device__ float2 operator+(const float2 &a, const float2 &b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator+(const float2 &a, const float s) {
	return make_float2(a.x + s, a.y + s);
}

__device__ float2 operator+(const float s, const float2 &a) {
	return a + s;
}

__device__ float2 operator*(const float2 &a, const float s) {
	return make_float2(a.x * s, a.y * s);
}

__device__ float2 operator*(const float s, const float2 &a) {
	return a * s;
}

__device__ float2 operator/(const float2 &a, const float2 &b) {
	return make_float2(a.x / b.x, a.y / b.y);
}

