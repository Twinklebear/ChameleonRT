#ifndef TROWBRIDGE_REITZ_MICROFACE_HLSL
#define TROWBRIDGE_REITZ_MICROFACE_HLSL

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

// TODO: Use the more accurate masking/shadowing function from PBRT?
// Or use the Smith approx as mentioned by Burley?
float gtr_2_masking_shadowing(in const DisneyMaterial mat,
                              in const Basis basis,
                              in const float3 w,
                              in const float3 w_h)
{
    if (dot(w, w_h) / dot(w, basis.z) <= 0) {
        return 0.f;
    }
    const float cos_theta_sqr = pow2(dot(w, basis.z));
    const float tan_theta_sqr = (1.f - cos_theta_sqr) / cos_theta_sqr;
    return 2.f / (1.f + sqrt(1.f + pow2(mat.roughness) * tan_theta_sqr));
}

#endif
