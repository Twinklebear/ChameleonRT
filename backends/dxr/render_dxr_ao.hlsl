#include "util.hlsl"
#include "lcg_rng.hlsl"

#define NUM_AO_SAMPLES 2

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

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

struct RayPayloadPrimary {
    float3 normal;
    float dist;
};

[shader("raygeneration")] 
void RayGen_AO() {
    const uint2 pixel = DispatchRaysIndex().xy;
    const float2 dims = float2(DispatchRaysDimensions().xy);
    LCGRand rng = get_rng(frame_id);
    const float2 d = (pixel + float2(lcg_randomf(rng), lcg_randomf(rng))) / dims;

    RayDesc ray;
    ray.Origin = cam_pos.xyz;
    ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
    ray.TMin = 0;
    ray.TMax = 1e20f;

    uint ray_count = 0;
    RayPayloadPrimary payload;
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, PRIMARY_RAY, 1, PRIMARY_RAY, ray, payload);
#ifdef REPORT_RAY_STATS
    ++ray_count;
#endif

    float3 ao_color = 0.f;
    if (payload.dist > 0.f) {
        float3 v_z = payload.normal;
        float3 v_x, v_y;
        ortho_basis(v_x, v_y, v_z);

        ray.Origin = ray.Origin + ray.Direction * payload.dist;
        ray.TMin = EPSILON;
        ray.TMax = 1e20f;

        float n_occluded = 0;
        OcclusionHitInfo shadow_hit;
        for (int i = 0; i < NUM_AO_SAMPLES; ++i) {
            const float theta = sqrt(lcg_randomf(rng));
            const float phi = 2.f * M_PI * lcg_randomf(rng);

            const float x = cos(phi) * theta;
            const float y = sin(phi) * theta;
            const float z = sqrt(1.f - theta * theta);

            ray.Direction = normalize(x * v_x + y * v_y + z * v_z);

            const uint32_t occlusion_flags = RAY_FLAG_FORCE_OPAQUE
                | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

            shadow_hit.hit = 1;
            TraceRay(scene, occlusion_flags, 0xff, PRIMARY_RAY, 1, OCCLUSION_RAY, ray, shadow_hit);

            if (shadow_hit.hit == 1) { 
                n_occluded += 1.f;
            }
        }
        ao_color = 1.f - n_occluded / NUM_AO_SAMPLES;
    }

    const float4 accum_color = (float4(ao_color, 1.0) + frame_id * accum_buffer[pixel]) / (frame_id + 1);
    accum_buffer[pixel] = accum_color;

    output[pixel] = float4(linear_to_srgb(accum_color.r),
            linear_to_srgb(accum_color.g),
            linear_to_srgb(accum_color.b), 1.f);

#ifdef REPORT_RAY_STATS
    ray_stats[pixel] = ray_count;
#endif
}

[shader("miss")]
void Miss_AO(inout RayPayloadPrimary payload : SV_RayPayload) {
    payload.normal = 0.f;
    payload.dist = -1.f;
}

[shader("miss")]
void ShadowMiss_AO(inout OcclusionHitInfo occlusion : SV_RayPayload) {
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
void ClosestHit_AO(inout RayPayloadPrimary payload, Attributes attrib) {
    uint3 idx = indices[NonUniformResourceIndex(PrimitiveIndex())];

    float3 va = vertices[NonUniformResourceIndex(idx.x)];
    float3 vb = vertices[NonUniformResourceIndex(idx.y)];
    float3 vc = vertices[NonUniformResourceIndex(idx.z)];
    float3 ng = normalize(cross(vb - va, vc - va));

    float3x3 inv_transp = float3x3(WorldToObject4x3()[0], WorldToObject4x3()[1], WorldToObject4x3()[2]);
    payload.normal = normalize(mul(inv_transp, ng));
    payload.dist = RayTCurrent();
}



