#include "util.hlsl"
#include "lcg_rng.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> output : register(u0);

// Accumulation buffer for progressive refinement
RWTexture2D<float4> accum_buffer : register(u1);

// View params buffer
cbuffer ViewParams : register(b0) {
    float4 cam_pos;
    float4 cam_du;
    float4 cam_dv;
    float4 cam_dir_top_left;
}

cbuffer FrameId : register(b2) {
    uint32_t frame_id;
}

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

struct RayPayloadNg {
    float3 color;
};

[shader("raygeneration")] 
void RayGen_NG() {
    const uint2 pixel = DispatchRaysIndex().xy;
    const float2 dims = float2(DispatchRaysDimensions().xy);
    LCGRand rng = get_rng(frame_id);

    RayPayloadNg payload;
    payload.color = float3(0.f, 0.f, 0.f);
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        const float2 d = (pixel + float2(lcg_randomf(rng), lcg_randomf(rng))) / dims;

        RayDesc ray;
        ray.Origin = cam_pos.xyz;
        ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
        ray.TMin = 0;
        ray.TMax = 1e20f;

        uint ray_count = 0;
        TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, PRIMARY_RAY, 1, PRIMARY_RAY, ray, payload);
    }
    payload.color /= NUM_SAMPLES;

    const float4 accum_color = (float4(payload.color, 1.0) + frame_id * accum_buffer[pixel]) / (frame_id + 1);
    accum_buffer[pixel] = accum_color;

    output[pixel] = float4(linear_to_srgb(accum_color.r),
            linear_to_srgb(accum_color.g),
            linear_to_srgb(accum_color.b), 1.f);
}

[shader("miss")]
void Miss_NG(inout RayPayloadNg payload : SV_RayPayload) {
    payload.color = 0.f;
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
void ClosestHit_NG(inout RayPayloadNg payload, Attributes attrib) {
    uint3 idx = indices[NonUniformResourceIndex(PrimitiveIndex())];

    float3 va = vertices[NonUniformResourceIndex(idx.x)];
    float3 vb = vertices[NonUniformResourceIndex(idx.y)];
    float3 vc = vertices[NonUniformResourceIndex(idx.z)];
    float3 ng = normalize(cross(vb - va, vc - va));

    float3x3 inv_transp = float3x3(WorldToObject4x3()[0], WorldToObject4x3()[1], WorldToObject4x3()[2]);
    ng = normalize(mul(inv_transp, ng));
    if (dot(ng, WorldRayDirection()) > 0.0) {
        ng = -ng;
    }
    payload.color += 0.5f * (ng + float3(1, 1, 1));
}


