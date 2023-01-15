#include "cuda_utils.h"

typedef void QuadLight;
#include "optix_params.h"

extern "C" {
__constant__ LaunchParams launch_params;
}

struct RayPayload {
    // payload registers 0, 1
    float2 uv;
    // payload register 2
    float t_hit;
};

__device__ RayPayload make_ray_payload()
{
    RayPayload p;
    p.uv = make_float2(0.f);
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
        const float2 d = make_float2(pixel.x + 0.5f, pixel.y + 0.5f) / make_float2(screen);
        float3 ray_dir = normalize(d.x * make_float3(launch_params.cam_du) +
                                   d.y * make_float3(launch_params.cam_dv) +
                                   make_float3(launch_params.cam_dir_top_left));

        float3 ray_origin = make_float3(launch_params.cam_pos);

        RayPayload payload = make_ray_payload();

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
                   reinterpret_cast<uint32_t &>(payload.uv.x),
                   reinterpret_cast<uint32_t &>(payload.uv.y),
                   reinterpret_cast<uint32_t &>(payload.t_hit));
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif

        if (payload.t_hit > 0.f) {
            illum = illum +
                    make_float3(payload.uv.x, payload.uv.y, 1.f - payload.uv.x - payload.uv.y);
        }
    }
    illum = illum / launch_params.samples_per_pixel;

    launch_params.accum_buffer[pixel_idx] = make_float4(illum, 1.f);

    launch_params.framebuffer[pixel_idx] =
        make_uchar4(clamp(linear_to_srgb(illum.x) * 255.f, 0.f, 255.f),
                    clamp(linear_to_srgb(illum.y) * 255.f, 0.f, 255.f),
                    clamp(linear_to_srgb(illum.z) * 255.f, 0.f, 255.f),
                    255);

#ifdef REPORT_RAY_STATS
    launch_params.ray_stats_buffer[pixel_idx] = ray_count;
#endif
}

extern "C" __global__ void __miss__miss()
{
    optixSetPayload_1(__float_as_int(-1.f));
}

extern "C" __global__ void __miss__occlusion_miss()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__closest_hit()
{
    const HitGroupParams &params = get_shader_params<HitGroupParams>();

    const float2 bary = optixGetTriangleBarycentrics();

    optixSetPayload_0(__float_as_int(bary.x));
    optixSetPayload_1(__float_as_int(bary.y));

    optixSetPayload_2(__float_as_int(optixGetRayTmax()));
}
