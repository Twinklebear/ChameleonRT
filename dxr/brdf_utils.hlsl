#ifndef BRDF_UTILS_HLSL
#define BRDF_UTILS_HLSL

#include "lcg_rng.hlsl"
#include "util.hlsl"

struct DisneyMaterial {
    float3 base_color;
    float metallic;

    float specular;
    float roughness;
    float specular_tint;
    float anisotropy;

    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;

    float ior;
    float specular_transmission;
};

bool same_hemisphere(in const float3 w_o, in const float3 w_i, in const float3 n)
{
    return dot(w_o, n) * dot(w_i, n) > 0.f;
}

// Sample the hemisphere using a cosine weighted distribution,
// returns a vector in a hemisphere oriented about (0, 0, 1)
float3 cos_sample_hemisphere(float2 u)
{
    float2 s = 2.f * u - 1.f;
    float2 d;
    float radius = 0;
    float theta = 0;
    if (s.x == 0.f && s.y == 0.f) {
        d = s;
    } else {
        if (abs(s.x) > abs(s.y)) {
            radius = s.x;
            theta = M_PI / 4.f * (s.y / s.x);
        } else {
            radius = s.y;
            theta = M_PI / 2.f - M_PI / 4.f * (s.x / s.y);
        }
    }
    d = radius * float2(cos(theta), sin(theta));
    return float3(d.x, d.y, sqrt(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
}

float3 spherical_dir(float sin_theta, float cos_theta, float phi)
{
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

float3 schlick_fresnel(const float3 r_0, const float cos_theta)
{
    return r_0 + (1.f - r_0) * pow(1.f - cos_theta, 5.f);
}

#endif

