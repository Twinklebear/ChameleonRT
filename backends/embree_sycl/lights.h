#pragma once

#include "float3.h"
#include "util.h"

namespace kernel {

float3 sample_quad_light_position(const QuadLight &light, float2 samples)
{
    // For simplicity of the initial port, just make float3 from the glm types
    const float3 light_v_x = make_float3(light.v_x.x, light.v_x.y, light.v_x.z);
    const float3 light_v_y = make_float3(light.v_y.x, light.v_y.y, light.v_y.z);
    const float3 light_position =
        make_float3(light.position.x, light.position.y, light.position.z);
    return samples.x * light_v_x * light.width + samples.y * light_v_y * light.height +
           light_position;
}

/* Compute the PDF of sampling the sampled point p light with the ray specified by orig and
 * dir, assuming the light is not occluded
 */
float quad_light_pdf(const QuadLight &light,
                     const float3 &p,
                     const float3 &orig,
                     const float3 &dir)
{
    const float3 light_normal = make_float3(light.normal.x, light.normal.y, light.normal.z);

    float surface_area = light.width * light.height;
    float3 to_pt = p - dir;
    float dist_sqr = dot(to_pt, to_pt);
    float n_dot_w = dot(light_normal, neg(dir));
    if (n_dot_w < EPSILON) {
        return 0.f;
    }
    return dist_sqr / (n_dot_w * surface_area);
}

bool quad_intersect(
    const QuadLight &light, const float3 &orig, const float3 &dir, float &t, float3 &light_pos)
{
    const float3 light_normal = make_float3(light.normal.x, light.normal.y, light.normal.z);
    float denom = dot(dir, light_normal);
    if (denom != 0.f) {
        const float3 light_v_x = make_float3(light.v_x.x, light.v_x.y, light.v_x.z);
        const float3 light_v_y = make_float3(light.v_y.x, light.v_y.y, light.v_y.z);
        const float3 light_position =
            make_float3(light.position.x, light.position.y, light.position.z);

        t = dot(light_position - orig, light_normal) / denom;
        if (t < 0.f) {
            return false;
        }

        // It's a finite plane so now see if the hit point is actually inside the plane
        light_pos = orig + dir * t;
        float3 hit_v = light_pos - light_position;
        if (sycl::fabs(dot(hit_v, light_v_x)) < light.width &&
            sycl::fabs(dot(hit_v, light_v_y)) < light.height) {
            return true;
        }
    }
    return false;
}

}
