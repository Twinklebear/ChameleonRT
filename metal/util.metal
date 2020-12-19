#pragma once

#include <metal_stdlib>
#include <simd/simd.h>

#define EPSILON 0.0001f

#define MAX_PATH_DEPTH 5

using namespace metal;

float linear_to_srgb(float x)
{
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

template <typename V>
float luminance(thread const V &c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float pow2(float x)
{
    return x * x;
}

void ortho_basis(thread float3 &v_x, thread float3 &v_y, thread const float3 &n)
{
    v_y = float3(0.f);

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

