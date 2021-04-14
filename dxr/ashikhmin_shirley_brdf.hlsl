#ifndef ASHIKHMIN_SHIRLEY_BRDF
#define ASHIKHMIN_SHIRLEY_BRDF

#include "lambertian_brdf.hlsl"

float gtr_2(in const DisneyMaterial mat,
            in const Basis basis,
            in const float3 w_o,
            in const float3 w_i,
            in const float3 w_h)
{
    const float alpha_sqr = pow2(mat.roughness);
    const float cos_theta_h = dot(w_h, basis.z);
    if (cos_theta_h == 0.f) {
        return 0.f;
    }
    return alpha_sqr * M_1_PI / pow2(1.f + (alpha_sqr - 1.f) * pow2(cos_theta_h));
}

float3 gtr_2_sample_h(in const DisneyMaterial mat, in const Basis basis, const float2 u)
{
    const float phi = 2.f * M_PI * u.x;
    const float cos_theta = sqrt((1.f - u.y) / (1.f + (pow2(mat.roughness) - 1.f) * u.y));
    const float sin_theta = sqrt(1.f - pow2(cos_theta));
    float3 w_h = normalize(spherical_dir(sin_theta, cos_theta, phi));
    return w_h.x * basis.x + w_h.y * basis.y + w_h.z * basis.z;
}

float gtr_2_pdf(in const DisneyMaterial mat,
                in const Basis basis,
                in const float3 w_o,
                in const float3 w_i,
                in const float3 w_h)
{
    return gtr_2(mat, basis, w_o, w_i, w_h) * abs(dot(w_h, basis.z));
}

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
    const float3 w_h = normalize(w_o + w_i);
    const float d = gtr_2(mat, basis, w_o, w_i, w_h);
    const float3 f = schlick_fresnel(r_spec, dot(w_o, basis.z));
    const float denom = 4.f * dot(w_h, w_i) * max(dot(w_o, basis.z), dot(w_i, basis.z));
    float3 specular = 0.f;
    if (denom != 0.f) {
        specular = d * f / denom;
    }

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
    float pdf = abs(dot(w_i, basis.z)) * M_1_PI;

    float3 w_h = normalize(w_o + w_i);
    const float p_h = gtr_2_pdf(mat, basis, w_o, w_i, w_h);
    const float o_dot_h = dot(w_o, w_h);
    if (o_dot_h != 0) {
        pdf += p_h / (4.f * o_dot_h);
    }
    return 0.5f * pdf;
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
    } else {
        // Sample direction from the microface component
        const float3 w_h = gtr_2_sample_h(mat, basis, u);
        w_i = reflect(w_o, w_h);
    }
    pdf = ashikhmin_shirley_pdf(mat, basis, w_o, w_i);
    return ashikhmin_shirley_brdf(mat, basis, w_o, w_i);
}

#endif

