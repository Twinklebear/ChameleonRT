#ifndef LAMBERTIAN_BRDF_HLSL
#define LAMBERTIAN_BRDF_HLSL

#include "brdf_utils.hlsl"

// Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using
// the random samples in s
float3 lambertian_sample_dir(in const Basis basis, in const float2 u)
{
    const float3 hemi_dir = normalize(cos_sample_hemisphere(u));
    return hemi_dir.x * basis.x + hemi_dir.y * basis.y + hemi_dir.z * basis.z;
}

float3 lambertian_brdf(in const DisneyMaterial mat,
                       in const Basis basis,
                       in const float3 w_o,
                       in const float3 w_i)
{
    return mat.base_color * M_1_PI;
}

float lambertian_pdf(in const DisneyMaterial mat,
                     in const Basis basis,
                     in const float3 w_o,
                     in const float3 w_i)
{
    if (same_hemisphere(w_o, w_i, basis.z)) {
        return abs(dot(w_i, basis.z)) * M_1_PI;
    }
    return 0.f;
}

float3 sample_lambertian_brdf(in const DisneyMaterial mat,
                              in const Basis basis,
                              in const float3 w_o,
                              inout LCGRand rng,
                              out float3 w_i,
                              out float pdf)
{
    const float2 u = float2(lcg_randomf(rng), lcg_randomf(rng));
    w_i = lambertian_sample_dir(basis, u);
    pdf = lambertian_pdf(mat, basis, w_o, w_i);
    return lambertian_brdf(mat, basis, w_o, w_i);
}

#endif
