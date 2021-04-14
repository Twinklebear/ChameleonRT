#ifndef TORRANCE_SPARROW_BRDF
#define TORRANCE_SPARROW_BRDF

float3 torrance_sparrow_brdf(in const DisneyMaterial mat,
                             in const Basis basis,
                             in const float3 w_o,
                             in const float3 w_i)
{
    const float cos_theta_o = abs(dot(w_o, basis.z));
    const float cos_theta_i = abs(dot(w_i, basis.z));
    if (cos_theta_o == 0.f || cos_theta_i == 0.f) {
        return 0.f;
    }

    float3 w_h = normalize(w_o + w_i);
    if (dot(w_h, basis.z) < 0.f) {
        w_h = -w_h;
    }
    const float d = gtr_2(mat, basis, w_o, w_i, w_h);
    const float g = gtr_2_masking_shadowing(mat, basis, w_o, w_h) *
                    gtr_2_masking_shadowing(mat, basis, w_i, w_h);
    // The light comes back along w_i and reflects off the surface to return
    // along w_o back to the eye. So w_i is the incident ray , and w_o is
    // the transmitted ray. Also note: Fresnel is computed vs. the microfacet normal,
    // not the macrosurface normal, as the ray is reflected by the microfacet.
    const float f = fresnel_dielectric(dot(w_i, w_h), mat.ior, 1.f);
    return mat.base_color * d * g * f / (4.f * cos_theta_o * cos_theta_i);
}

float torrance_sparrow_pdf(in const DisneyMaterial mat,
                           in const Basis basis,
                           in const float3 w_o,
                           in const float3 w_i)
{
    if (!same_hemisphere(w_o, w_i, basis.z)) {
        return 0.f;
    }

    const float3 w_h = normalize(w_o + w_i);
    const float p_h = gtr_2_pdf(mat, basis, w_o, w_i, w_h);
    return p_h / (4.f * dot(w_o, w_h));
}

float3 sample_torrance_sparrow_brdf(in const DisneyMaterial mat,
                                    in const Basis basis,
                                    in const float3 w_o,
                                    inout LCGRand rng,
                                    out float3 w_i,
                                    out float pdf)
{
    float2 u = float2(lcg_randomf(rng), lcg_randomf(rng));
    const float3 w_h = gtr_2_sample_h(mat, basis, u);
    w_i = reflect(w_o, w_h);
    if (!same_hemisphere(w_o, w_i, basis.z)) {
        pdf = 0.f;
        return 0.f;
    }
    pdf = gtr_2_pdf(mat, basis, w_o, w_i, w_h) / (4.f * dot(w_o, w_h));
    return torrance_sparrow_brdf(mat, basis, w_o, w_i);
}

#endif

