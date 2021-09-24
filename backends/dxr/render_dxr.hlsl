#include "util.hlsl"
#include "lcg_rng.hlsl"
#include "disney_bsdf.hlsl"
#include "lights.hlsl"
#include "util/texture_channel_mask.h"

struct MaterialParams {
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
    float2 pad;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> output : register(u0);

// Accumulation buffer for progressive refinement
RWTexture2D<float4> accum_buffer : register(u1);

#ifdef REPORT_RAY_STATS
RWTexture2D<uint> ray_stats : register(u2);
#endif

// View params buffer
cbuffer ViewParams : register(b0) {
    float4 cam_pos;
    float4 cam_du;
    float4 cam_dv;
    float4 cam_dir_top_left;
    uint32_t frame_id;
}

cbuffer SceneParams : register(b1) {
    uint32_t num_lights;
};

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

StructuredBuffer<MaterialParams> material_params : register(t1);

StructuredBuffer<QuadLight> lights : register(t2);

Texture2D textures[] : register(t3);
SamplerState tex_sampler : register(s0);

float textured_scalar_param(const float x, in const float2 uv) {
    const uint32_t mask = asuint(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return textures[NonUniformResourceIndex(tex_id)]
            .SampleLevel(tex_sampler, uv, 0)[channel];
    }
    return x;
}


void unpack_material(inout DisneyMaterial mat, uint id, float2 uv) {
    MaterialParams p = material_params[NonUniformResourceIndex(id)];

    const uint32_t mask = asuint(p.base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color = textures[NonUniformResourceIndex(tex_id)]
            .SampleLevel(tex_sampler, uv, 0).xyz;
    } else {
        mat.base_color = p.base_color;
    }

    mat.metallic = textured_scalar_param(p.metallic, uv);
    mat.specular = textured_scalar_param(p.specular, uv);
    mat.roughness = textured_scalar_param(p.roughness, uv);
    mat.specular_tint = textured_scalar_param(p.specular_tint, uv);
    mat.anisotropy = textured_scalar_param(p.anisotropy, uv);
    mat.sheen = textured_scalar_param(p.sheen, uv);
    mat.sheen_tint = textured_scalar_param(p.sheen_tint, uv);
    mat.clearcoat = textured_scalar_param(p.clearcoat, uv);
    mat.clearcoat_gloss = textured_scalar_param(p.clearcoat_gloss, uv);
    mat.ior = textured_scalar_param(p.ior, uv);
    mat.specular_transmission = textured_scalar_param(p.specular_transmission, uv);
}

float3 sample_direct_light(in const DisneyMaterial mat, in const float3 hit_p, in const float3 n,
        in const float3 v_x, in const float3 v_y, in const float3 w_o, inout uint ray_count, inout LCGRand rng)
{
    float3 illum = 0.f;

    uint32_t light_id = lcg_randomf(rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    QuadLight light = lights[NonUniformResourceIndex(light_id)];

    OcclusionHitInfo shadow_hit;
    RayDesc shadow_ray;
    shadow_ray.Origin = hit_p;
    shadow_ray.TMin = EPSILON;

    const uint32_t occlusion_flags = RAY_FLAG_FORCE_OPAQUE
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
        | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos = sample_quad_light_position(light,
                float2(lcg_randomf(rng), lcg_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        shadow_hit.hit = 1;
        shadow_ray.Direction = light_dir;
        shadow_ray.TMax = light_dist;
        TraceRay(scene, occlusion_flags, 0xff, PRIMARY_RAY, 1, OCCLUSION_RAY, shadow_ray, shadow_hit);
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_hit.hit == 0) {
            float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
            float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
            illum = bsdf * light.emission.rgb * abs(dot(light_dir, n)) * w / light_pdf;
        }
    }

    // Sample the BRDF to compute a light sample as well
    {
        float3 w_i;
        float bsdf_pdf;
        float3 bsdf = sample_disney_brdf(mat, n, w_o, v_x, v_y, rng, w_i, bsdf_pdf);

        float light_dist;
        float3 light_pos;
        if (any(bsdf > 0.f) && bsdf_pdf >= EPSILON
                && quad_intersect(light, hit_p, w_i, light_dist, light_pos))
        {
            float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
            if (light_pdf >= EPSILON) {
                float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
                shadow_hit.hit = 1;
                shadow_ray.Direction = w_i;
                shadow_ray.TMax = light_dist;
                TraceRay(scene, occlusion_flags, 0xff, PRIMARY_RAY, 1, OCCLUSION_RAY, shadow_ray, shadow_hit);
#ifdef REPORT_RAY_STATS
                ++ray_count;
#endif
                if (shadow_hit.hit == 0) {
                    illum += bsdf * light.emission.rgb * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

[shader("raygeneration")] 
void RayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    const float2 dims = float2(DispatchRaysDimensions().xy);
    LCGRand rng = get_rng(frame_id);
    const float2 d = (pixel + float2(lcg_randomf(rng), lcg_randomf(rng))) / dims;

    RayDesc ray;
    ray.Origin = cam_pos.xyz;
    ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
    ray.TMin = 0;
    ray.TMax = 1e20f;

    DisneyMaterial mat;

    uint ray_count = 0;
    int bounce = 0;
    float3 illum = float3(0, 0, 0);
    float3 path_throughput = float3(1, 1, 1);
    do {
        HitInfo payload;
        payload.color_dist = float4(0, 0, 0, -1);
        TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, PRIMARY_RAY, 1, PRIMARY_RAY, ray, payload);
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif

        // If we hit nothing, include the scene background color from the miss shader
        if (payload.color_dist.w <= 0) {
            illum += path_throughput * payload.color_dist.rgb;
            break;
        }

        const float3 w_o = -ray.Direction;
        const float3 hit_p = ray.Origin + payload.color_dist.w * ray.Direction;
        unpack_material(mat, uint(payload.normal.w), payload.color_dist.rg);

        float3 v_x, v_y;
        float3 v_z = payload.normal.xyz;
        // For opaque objects (or in the future, thin ones) make the normal face forward
        if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.0) {
            v_z = -v_z;
        }
        ortho_basis(v_x, v_y, v_z);

        illum += path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o, ray_count, rng);

        float3 w_i;
        float pdf;
        float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
        if (pdf == 0.f || all(bsdf == 0.f)) {
            break;
        }
        path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

        ray.Origin = hit_p;
        ray.Direction = w_i;
        ray.TMin = EPSILON;
        ray.TMax = 1e20f;
        ++bounce;

        // Russian roulette termination
        if (bounce > 3) {
            const float q = max(0.05f, 1.f - max(path_throughput.x, max(path_throughput.y, path_throughput.z)));
            if (lcg_randomf(rng) < q) {
                break;
            }
            path_throughput = path_throughput / (1.f - q);
        }
    } while (bounce < MAX_PATH_DEPTH);

    const float4 accum_color = (float4(illum, 1.0) + frame_id * accum_buffer[pixel]) / (frame_id + 1);
    accum_buffer[pixel] = accum_color;

    output[pixel] = float4(linear_to_srgb(accum_color.r),
            linear_to_srgb(accum_color.g),
            linear_to_srgb(accum_color.b), 1.f);

#ifdef REPORT_RAY_STATS
    ray_stats[pixel] = ray_count;
#endif
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload) {
    payload.color_dist.w = -1.f;

    float3 dir = WorldRayDirection();
    float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
    float v = acos(dir.y) * M_1_PI;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
        payload.color_dist.rgb = 0.5f;
    } else {
        payload.color_dist.rgb = 0.1f;
    }
}

[shader("miss")]
void ShadowMiss(inout OcclusionHitInfo occlusion : SV_RayPayload) {
    occlusion.hit = 0;
}

// Per-mesh parameters for the closest hit
StructuredBuffer<float3> vertices : register(t0, space1);
StructuredBuffer<uint3> indices : register(t1, space1);
StructuredBuffer<float3> normals : register(t2, space1);
StructuredBuffer<float2> uvs : register(t3, space1);

cbuffer MeshData : register(b0, space1) {
    uint32_t num_normals;
    uint32_t num_uvs;
    uint32_t material_id;
}

[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) {
    uint3 idx = indices[NonUniformResourceIndex(PrimitiveIndex())];

    float3 va = vertices[NonUniformResourceIndex(idx.x)];
    float3 vb = vertices[NonUniformResourceIndex(idx.y)];
    float3 vc = vertices[NonUniformResourceIndex(idx.z)];
    float3 ng = normalize(cross(vb - va, vc - va));

    float3 n = ng;
    // TODO per-vertex normals seems to give some pretty bad silhouetting,
    // even on some models which do seem well tesselated?
    // Implement shadow/bump terminator fix?
#if 0
    if (num_normals > 0) {
        float3 na = normals[NonUniformResourceIndex(idx.x)];
        float3 nb = normals[NonUniformResourceIndex(idx.y)];
        float3 nc = normals[NonUniformResourceIndex(idx.z)];
        n = normalize((1.f - attrib.bary.x - attrib.bary.y) * na
                + attrib.bary.x * nb + attrib.bary.y * nc);
    }
#endif

    float2 uv = float2(0, 0);
    if (num_uvs > 0) {
        float2 uva = uvs[NonUniformResourceIndex(idx.x)];
        float2 uvb = uvs[NonUniformResourceIndex(idx.y)];
        float2 uvc = uvs[NonUniformResourceIndex(idx.z)];
        uv = (1.f - attrib.bary.x - attrib.bary.y) * uva
            + attrib.bary.x * uvb + attrib.bary.y * uvc;
    }

    payload.color_dist = float4(uv, 0, RayTCurrent());
    float3x3 inv_transp = float3x3(WorldToObject4x3()[0], WorldToObject4x3()[1], WorldToObject4x3()[2]);
    payload.normal = float4(normalize(mul(inv_transp, n)), material_id);
}

