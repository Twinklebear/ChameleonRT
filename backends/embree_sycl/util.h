#pragma once

#include "float3.h"

#define M_PI_F 3.14159265358979323846f
#define M_1_PI_F 0.318309886183790671538f
#define EPSILON 0.0001f

#define MAX_PATH_DEPTH 5

namespace kernel {

float linear_to_srgb(float x)
{
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * sycl::native::powr(x, 1.f / 2.4f) - 0.055f;
}

float luminance(const float3 &c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float pow2(float x)
{
    return x * x;
}

uint32_t intbits(const float x)
{
    return sycl::bit_cast<uint32_t>(x);
}

void ortho_basis(float3 &v_x, float3 &v_y, const float3 &n)
{
    v_y = make_float3(0.f);

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

// Utils from HLSL
float saturate(float x)
{
    return sycl::min(sycl::max(x, 0.f), 1.f);
}

float lerp(float x, float y, float s)
{
    return x * (1.f - s) + y * s;
}

float3 lerp(float3 x, float3 y, float s)
{
    return x * (1.f - s) + y * s;
}

float3 reflect(const float3 &i, const float3 &n)
{
    return i - 2.f * n * dot(i, n);
}

float3 refract(const float3 &i, const float3 &n, float eta)
{
    float n_dot_i = dot(n, i);
    float k = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
    if (k < 0.f) {
        return make_float3(0.f);
    }
    return eta * i - (eta * n_dot_i + sycl::native::sqrt(k)) * n;
}

void set_ray(RTCRay &ray, const float3 &pos, const float3 &dir, const float tnear)
{
    ray.org_x = pos.x;
    ray.org_y = pos.y;
    ray.org_z = pos.z;
    ray.tnear = tnear;

    ray.dir_x = dir.x;
    ray.dir_y = dir.y;
    ray.dir_z = dir.z;
    ray.time = 0.f;
    ray.tfar = 1e20f;

    ray.mask = -1;
    ray.id = 0;
    ray.flags = 0;
}

RTCRay make_ray(const float3 &pos, const float3 &dir, const float tnear)
{
    RTCRay ray;
    set_ray(ray, pos, dir, tnear);
    return ray;
}

void set_ray_hit(RTCRayHit &ray_hit, const float3 &pos, const float3 &dir, const float tnear)
{
    ray_hit.ray.org_x = pos.x;
    ray_hit.ray.org_y = pos.y;
    ray_hit.ray.org_z = pos.z;
    ray_hit.ray.tnear = tnear;

    ray_hit.ray.dir_x = dir.x;
    ray_hit.ray.dir_y = dir.y;
    ray_hit.ray.dir_z = dir.z;
    ray_hit.ray.time = 0.f;
    ray_hit.ray.tfar = 1e20f;

    ray_hit.ray.mask = -1;
    ray_hit.ray.id = 0;
    ray_hit.ray.flags = 0;

    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
}

RTCRayHit make_ray_hit(const float3 &pos, const float3 &dir, const float tnear)
{
    RTCRayHit ray_hit;
    set_ray_hit(ray_hit, pos, dir, tnear);
    return ray_hit;
};

}
