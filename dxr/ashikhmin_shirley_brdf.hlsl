#ifndef ASHIKHMIN_SHIRLEY_BRDF
#define ASHIKHMIN_SHIRLEY_BRDF

#include "lambertian_brdf.hlsl"
#include "trowbridge_reitz_microfacet.hlsl"

float3 ashikhmin_shirley_brdf(in const DisneyMaterial mat,
                              in const Basis basis,
                              in const float3 w_o,
                              in const float3 w_i)
{
    const float3 r_dif = mat.base_color;
    const float3 r_spec =
        mat.specular * lerp(float3(1.f, 1.f, 1.f), mat.base_color, mat.specular_tint);

    // Diffuse term
    const float3 diffuse = M_1_PI * (28.f * r_dif / 23.f) * (1.f - r_spec) *
                           (1 - pow(1 - dot(w_i, basis.z) * 0.5f, 5.f)) *
                           (1 - pow(1 - dot(w_o, basis.z) * 0.5f, 5.f));

    // Specular term
    float3 w_h = w_o + w_i;
    if (all(w_h == float3(0.f, 0.f, 0.f))) {
        return 0.f;
    }
    w_h = normalize(w_h);
    const float d = gtr_2(mat, basis, w_o, w_i, w_h);
    // TODO: in pbrt-v3 they use dot(w_i, basis.z) (i.e., CosTheta(w_i) in their
    // coordinate system. Why? It makes some sense b/c w_i is the incident
    // light direction, but in the end it gives the wrong result?
    const float3 f = schlick_fresnel(r_spec, dot(w_o, basis.z));
    const float denom = 4.f * dot(w_h, w_i) * max(dot(w_o, basis.z), dot(w_i, basis.z));
    float3 specular = (d * f) / denom;

    return specular + diffuse;
}

float ashikhmin_shirley_pdf(in const DisneyMaterial mat,
                            in const Basis basis,
                            in const float3 w_o,
                            in const float3 w_i)
{
    if (!same_hemisphere(w_o, w_i, basis.z)) {
        return 0.f;
    }

    const float3 w_h = normalize(w_o + w_i);
    const float p_h = gtr_2_pdf(mat, basis, w_o, w_i, w_h);
    return 0.5f * (abs(dot(w_i, basis.z)) * M_1_PI + p_h / (4.f * dot(w_o, w_h)));
}

float3 sample_ashikhmin_shirley_brdf(in const DisneyMaterial mat,
                                     in const Basis basis,
                                     in const float3 w_o,
                                     inout LCGRand rng,
                                     out float3 w_i,
                                     out float pdf)
{
    float comp = lcg_randomf(rng);
    float2 u = float2(lcg_randomf(rng), lcg_randomf(rng));
    // Pick between the two components of the model
    if (comp < 0.5) {
        // Sample direction from the diffuse component
        w_i = lambertian_sample_dir(basis, u);
        if (!same_hemisphere(w_o, w_i, basis.z)) {
            w_i = flip_hemisphere(basis, w_i);
        }
    } else {
        // Sample direction from the microface component
        const float3 w_h = gtr_2_sample_h(mat, basis, u);
        w_i = reflect(w_o, w_h);
        if (!same_hemisphere(w_o, w_i, basis.z)) {
            return 0.f;
        }
    }
    pdf = ashikhmin_shirley_pdf(mat, basis, w_o, w_i);
    return ashikhmin_shirley_brdf(mat, basis, w_o, w_i);
}

#endif

