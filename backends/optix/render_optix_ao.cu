#include "cuda_utils.h"
#include "lcg_rng.h"

typedef void QuadLight;
#include "optix_params.h"

#define NUM_AO_SAMPLES 4

extern "C" {
__constant__ LaunchParams launch_params;
}

struct RayPayload {
    // payload registers 0, 1, 2
    float3 normal;
    // payload register 3
    float t_hit;
};

struct AOPayload {
    // payload register 0
    uint32_t n_occluded;
};

__device__ RayPayload make_ray_payload()
{
    RayPayload p;
    p.normal = make_float3(0.f);
    p.t_hit = -1.f;
    return p;
}

extern "C" __global__ void __raygen__perspective_camera()
{
    const RayGenParams &params = get_shader_params<RayGenParams>();

    const uint2 pixel = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint2 screen =
        make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
    const uint32_t pixel_idx = pixel.x + pixel.y * screen.x;

    uint16_t ray_count = 0;
    float3 illum = make_float3(0.f);
    // Keep spp for benchmarking, but we don't use RNG at all here to minimize kernel size
    for (uint32_t s = 0; s < launch_params.samples_per_pixel; ++s) {
        LCGRand rng = get_rng(launch_params.frame_id * launch_params.samples_per_pixel + s);
        const float2 d = make_float2(pixel.x + lcg_randomf(rng), pixel.y + lcg_randomf(rng)) /
                         make_float2(screen);
        float3 ray_dir = normalize(d.x * make_float3(launch_params.cam_du) +
                                   d.y * make_float3(launch_params.cam_dv) +
                                   make_float3(launch_params.cam_dir_top_left));

        float3 ray_origin = make_float3(launch_params.cam_pos);

        RayPayload payload = make_ray_payload();

#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif
        optixTrace(launch_params.scene,
                   ray_origin,
                   ray_dir,
                   EPSILON,
                   1e20f,
                   0.f,
                   0xff,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   PRIMARY_RAY,
                   1,
                   PRIMARY_RAY,
                   reinterpret_cast<uint32_t &>(payload.normal.x),
                   reinterpret_cast<uint32_t &>(payload.normal.y),
                   reinterpret_cast<uint32_t &>(payload.normal.z),
                   reinterpret_cast<uint32_t &>(payload.t_hit));

        if (payload.t_hit > 0.f) {
            ray_origin = ray_origin + ray_dir * payload.t_hit;

            float3 v_x, v_y;
            float3 v_z = payload.normal;
            if (dot(ray_dir, v_z) > 0.f) {
                v_z = -v_z;
            }
            ortho_basis(v_x, v_y, v_z);

            const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                                             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                                             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
            AOPayload ao_payload;
            ao_payload.n_occluded = NUM_AO_SAMPLES;
#ifdef REPORT_RAY_STATS
            ray_count += NUM_AO_SAMPLES;
#endif
            for (int i = 0; i < NUM_AO_SAMPLES; ++i) {
                const float theta = sqrt(lcg_randomf(rng));
                const float phi = 2.f * M_PI * lcg_randomf(rng);

                const float x = cos(phi) * theta;
                const float y = sin(phi) * theta;
                const float z = sqrt(1.f - theta * theta);

                ray_dir = normalize(x * v_x + y * v_y + z * v_z);

                optixTrace(launch_params.scene,
                           ray_origin,
                           ray_dir,
                           EPSILON,
                           1e20f,
                           0.f,
                           0xff,
                           occlusion_flags,
                           PRIMARY_RAY,
                           1,
                           OCCLUSION_RAY,
                           ao_payload.n_occluded);
            }
            illum = illum + 1.f - make_float3(ao_payload.n_occluded) / NUM_AO_SAMPLES;
        }
    }
    illum = illum / launch_params.samples_per_pixel;

    const float3 prev_color = make_float3(launch_params.accum_buffer[pixel_idx]);
    const float3 accum_color =
        (illum + launch_params.frame_id * prev_color) / (launch_params.frame_id + 1);
    launch_params.accum_buffer[pixel_idx] = make_float4(accum_color, 1.f);

    launch_params.framebuffer[pixel_idx] =
        make_uchar4(clamp(linear_to_srgb(accum_color.x) * 255.f, 0.f, 255.f),
                    clamp(linear_to_srgb(accum_color.y) * 255.f, 0.f, 255.f),
                    clamp(linear_to_srgb(accum_color.z) * 255.f, 0.f, 255.f),
                    255);

#ifdef REPORT_RAY_STATS
    launch_params.ray_stats_buffer[pixel_idx] = ray_count;
#endif
}

extern "C" __global__ void __miss__miss()
{
    optixSetPayload_3(__float_as_int(-1.f));
}

extern "C" __global__ void __miss__occlusion_miss()
{
    uint32_t n_occluded = optixGetPayload_0();
    optixSetPayload_0(n_occluded - 1);
}

extern "C" __global__ void __closesthit__closest_hit()
{
    const HitGroupParams &params = get_shader_params<HitGroupParams>();

    const float2 bary = optixGetTriangleBarycentrics();
    const uint3 indices = params.index_buffer[optixGetPrimitiveIndex()];
    const float3 v0 = params.vertex_buffer[indices.x];
    const float3 v1 = params.vertex_buffer[indices.y];
    const float3 v2 = params.vertex_buffer[indices.z];
    float3 normal = normalize(cross(v1 - v0, v2 - v0));
    normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

    optixSetPayload_0(__float_as_int(normal.x));
    optixSetPayload_1(__float_as_int(normal.y));
    optixSetPayload_2(__float_as_int(normal.z));

    optixSetPayload_3(__float_as_int(optixGetRayTmax()));
}

