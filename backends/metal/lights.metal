#pragma once

#include <metal_stdlib>
#include <simd/simd.h>
#include "util.metal"

using namespace metal;

// Quad-shaped light source
struct QuadLight {
    float4 emission;
    float4 position;
    float4 normal;
    // x and y vectors spanning the quad, with
    // the half-width and height in the w component
    float4 v_x;
    float4 v_y;
};

float3 sample_quad_light_position(thread const QuadLight &light, float2 samples)
{
    return samples.x * light.v_x.xyz * light.v_x.w + samples.y * light.v_y.xyz * light.v_y.w +
           light.position.xyz;
}

/* Compute the PDF of sampling the sampled point p light with the ray specified by orig and
 * dir, assuming the light is not occluded
 */
float quad_light_pdf(thread const QuadLight &light,
                     thread const float3 &p,
                     thread const float3 &orig,
                     thread const float3 &dir)
{
    float surface_area = light.v_x.w * light.v_y.w;
    float3 to_pt = p - dir;
    float dist_sqr = dot(to_pt, to_pt);
    float n_dot_w = dot(light.normal.xyz, -dir);
    if (n_dot_w < EPSILON) {
        return 0.f;
    }
    return dist_sqr / (n_dot_w * surface_area);
}

bool quad_intersect(thread const QuadLight &light,
                    thread const float3 &orig,
                    thread const float3 &dir,
                    thread float &t,
                    thread float3 &light_pos)
{
    float denom = dot(dir, light.normal.xyz);
    if (denom != 0.f) {
        t = dot(light.position.xyz - orig, light.normal.xyz) / denom;
        if (t < 0.f) {
            return false;
        }

        // It's a finite plane so now see if the hit point is actually inside the plane
        light_pos = orig + dir * t;
        float3 hit_v = light_pos - light.position.xyz;
        if (abs(dot(hit_v, light.v_x.xyz)) < light.v_x.w &&
            abs(dot(hit_v, light.v_y.xyz)) < light.v_y.w) {
            return true;
        }
    }
    return false;
}

