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

// Flip the hemisphere that the v lies in relative to the basis z axis.
// I.e., flip the z component of v within the given basis
// Note: negate, or just flip the z component?
float3 flip_hemisphere(in const Basis basis, in const float3 v)
{
    return dot(v, basis.x) * basis.x + dot(v, basis.y) * basis.y - dot(v, basis.z) * basis.z;
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
    // Avoid precision issues with pow() here
    const float x = 1.f - cos_theta;
    return r_0 + (1.f - r_0) * x * x * x * x * x;
}

// Compute the Fresnel equation for dielectrics given the cosines of the
// incident & transmitted rays and the IORs of the materials on the incident
// and transmitted sides.
// cos_theta_inc: cos theta for the ray incident on the surface
// eta_inc: cos theta for the material on the indicent ray's side
// eta_tr: cos theta for the material on the transmitted ray's side
float fresnel_dielectric(float cos_theta_inc, const float eta_inc, const float eta_tr)
{
    cos_theta_inc = clamp(cos_theta_inc, -1.f, 1.f);
    const float sin_theta_inc = sqrt(max(0.f, 1.f - pow2(cos_theta_inc)));
    const float sin_theta_tr = eta_inc / eta_tr * sin_theta_inc;
    // Total internal reflection
    if (sin_theta_tr >= 1.f) {
        return 1.f;
    }

    const float cos_theta_tr = sqrt(max(0.f, 1.f - pow2(sin_theta_tr)));

    const float r_par = (eta_tr * cos_theta_inc - eta_inc * cos_theta_tr) /
                        (eta_tr * cos_theta_inc + eta_inc * cos_theta_tr);
    const float r_perp = (eta_inc * cos_theta_inc - eta_tr * cos_theta_tr) /
                         (eta_inc * cos_theta_inc + eta_tr * cos_theta_tr);
    return 0.5 * (pow2(r_par) + pow2(r_perp));
}

#endif

