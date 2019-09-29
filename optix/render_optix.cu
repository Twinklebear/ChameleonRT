#include "cuda_utils.h"
#include "pcg_rng.h"
#include "disney_bsdf.h"
#include "lights.h"
#include "optix_params.h"

extern "C" {
    __constant__ LaunchParams launch_params;
}

struct RayPayload {
    float2 uv;
    float t_hit;
    uint32_t material_id;

    float3 normal;
    float pad;
};

__device__ RayPayload make_ray_payload() {
    RayPayload p;
    p.uv = make_float2(0.f);
    p.t_hit = -1.f;
    p.material_id = 0;
    p.normal = make_float3(0.f);
    return p;
}

__device__ void unpack_material(const MaterialParams &p, float2 uv, DisneyMaterial &mat) {
    if (p.has_color_tex) {
        mat.base_color = make_float3(tex2D<float4>(p.color_texture, uv.x, uv.y));
    } else {
        mat.base_color = p.base_color;
    }

    mat.metallic = p.metallic;
    mat.specular = p.specular;
    mat.roughness = p.roughness;
    mat.specular_tint = p.specular_tint;
    mat.anisotropy = p.anisotropy;
    mat.sheen = p.sheen;
    mat.sheen_tint = p.sheen_tint;
    mat.clearcoat = p.clearcoat;
    mat.clearcoat_gloss = p.clearcoat_gloss;
    mat.ior = p.ior;
    mat.specular_transmission = p.specular_transmission;
}

__device__ float3 sample_direct_light(const DisneyMaterial &mat, const float3 &hit_p,
        const float3 &n, const float3 &v_x, const float3 &v_y, const float3 &w_o,
        const QuadLight *lights, const uint32_t num_lights, uint16_t &ray_count, PCGRand &rng)
{
    float3 illum = make_float3(0.f);

    uint32_t light_id = pcg32_randomf(rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    QuadLight light = lights[light_id];

    RayPayload shadow_payload = make_ray_payload();
    uint2 payload_ptr;
    pack_ptr(&shadow_payload, payload_ptr.x, payload_ptr.y);

    const uint32_t occlusion_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos = sample_quad_light_position(light, make_float2(pcg32_randomf(rng), pcg32_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        shadow_payload.t_hit = 1.f;
        optixTrace(launch_params.scene, hit_p, light_dir, EPSILON, light_dist, 0,
                0xff, occlusion_flags, PRIMARY_RAY, 1, OCCLUSION_RAY,
                payload_ptr.x, payload_ptr.y);
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_payload.t_hit <= 0.f) {
            float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
            float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
            illum = bsdf * light.emission * abs(dot(light_dir, n)) * w / light_pdf;
        }
    }

    // Sample the BRDF to compute a light sample as well
    {
        float3 w_i;
        float bsdf_pdf;
        float3 bsdf = sample_disney_brdf(mat, n, w_o, v_x, v_y, rng, w_i, bsdf_pdf);

        float light_dist;
        float3 light_pos;
        if (!all_zero(bsdf) && bsdf_pdf >= EPSILON && quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
            float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
            if (light_pdf >= EPSILON) {
                float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
                shadow_payload.t_hit = 1.f;
                optixTrace(launch_params.scene, hit_p, w_i, EPSILON, light_dist, 0,
                        0xff, occlusion_flags, PRIMARY_RAY, 1, OCCLUSION_RAY,
                        payload_ptr.x, payload_ptr.y);
#ifdef REPORT_RAY_STATS
                ++ray_count;
#endif
                if (shadow_payload.t_hit <= 0.f) {
                    illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

extern "C" __global__ void __raygen__perspective_camera() {
    const RayGenParams &params = get_shader_params<RayGenParams>();

    const uint2 pixel = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    const uint2 screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
    const uint32_t pixel_idx = pixel.x + pixel.y * screen.x;

    PCGRand rng = get_rng(pixel_idx * (launch_params.frame_id + 1));
    const float2 d = make_float2(pixel.x + pcg32_randomf(rng), pixel.y + pcg32_randomf(rng)) / make_float2(screen);
    float3 ray_dir = normalize(d.x * make_float3(launch_params.cam_du)
            + d.y * make_float3(launch_params.cam_dv) + make_float3(launch_params.cam_dir_top_left));

    float3 ray_origin = make_float3(launch_params.cam_pos);

    DisneyMaterial mat;

    uint16_t ray_count = 0;
    const float3 light_emission = make_float3(1.0);
    int bounce = 0;
    float3 illum = make_float3(0.0);
    float3 path_throughput = make_float3(1.0);
    do {
        RayPayload payload = make_ray_payload();
        uint2 payload_ptr;
        pack_ptr(&payload, payload_ptr.x, payload_ptr.y);

        optixTrace(launch_params.scene, ray_origin, ray_dir, EPSILON, 1e20f, 0,
                0xff, OPTIX_RAY_FLAG_DISABLE_ANYHIT, PRIMARY_RAY, 1, PRIMARY_RAY,
                payload_ptr.x, payload_ptr.y);
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif

        if (payload.t_hit <= 0.f) {
            illum = illum + path_throughput * payload.normal;
            break;
        }

        unpack_material(params.materials[payload.material_id], payload.uv, mat);

        const float3 w_o = -ray_dir;
        const float3 hit_p = ray_origin + payload.t_hit * ray_dir;
        float3 v_x, v_y;
        float3 v_z = payload.normal;
        if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.0) {
            v_z = -v_z;
        }
        ortho_basis(v_x, v_y, v_z);

        illum = illum + path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o,
                params.lights, params.num_lights, ray_count, rng);

        float3 w_i;
        float pdf;
        float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
        if (pdf < EPSILON || all_zero(bsdf)) {
            break;
        }
        path_throughput = path_throughput * bsdf * abs(dot(w_i, v_z)) / pdf;

        if (path_throughput.x < EPSILON && path_throughput.y < EPSILON && path_throughput.z < EPSILON) {
            break;
        }

        ray_origin = hit_p;
        ray_dir = w_i;

        ++bounce;
    } while (bounce < MAX_PATH_DEPTH);

    const float3 prev_color = make_float3(launch_params.accum_buffer[pixel_idx]);
    const float3 accum_color = (illum + launch_params.frame_id * prev_color) / (launch_params.frame_id + 1);
    launch_params.accum_buffer[pixel_idx] = make_float4(accum_color, 1.f);

    launch_params.framebuffer[pixel_idx] = make_uchar4(
            clamp(linear_to_srgb(accum_color.x) * 255.f, 0.f, 255.f),
            clamp(linear_to_srgb(accum_color.y) * 255.f, 0.f, 255.f),
            clamp(linear_to_srgb(accum_color.z) * 255.f, 0.f, 255.f), 255);

#ifdef REPORT_RAY_STATS
    launch_params.ray_stats_buffer[pixel_idx] = ray_count;
#endif
}

extern "C" __global__ void __miss__miss() {
    RayPayload &payload = get_payload<RayPayload>();
    payload.t_hit = -1.f;
    float3 dir = optixGetWorldRayDirection();
    // Apply our miss "shader" to draw the checkerboard background
    float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
    float v = acos(dir.y) * M_1_PI;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
        payload.normal = make_float3(0.5f);
    } else {
        payload.normal = make_float3(0.1f);
    }
}

extern "C" __global__ void __miss__occlusion_miss() {
    RayPayload &payload = get_payload<RayPayload>();
    payload.t_hit = -1.f;
}

extern "C" __global__ void __closesthit__closest_hit() {
    const HitGroupParams &params = get_shader_params<HitGroupParams>();

    const float2 bary = optixGetTriangleBarycentrics();
    const uint3 indices = params.index_buffer[optixGetPrimitiveIndex()];
    const float3 v0 = params.vertex_buffer[indices.x];
    const float3 v1 = params.vertex_buffer[indices.y];
    const float3 v2 = params.vertex_buffer[indices.z];
    const float3 normal = normalize(cross(v1 - v0, v2 - v0));

    float2 uv = make_float2(0.f);
    if (params.uv_buffer) {
        float2 uva = params.uv_buffer[indices.x];
        float2 uvb = params.uv_buffer[indices.y];
        float2 uvc = params.uv_buffer[indices.z];
        uv = (1.f - bary.x - bary.y) * uva
            + bary.x * uvb + bary.y * uvc;
    }

    RayPayload &payload = get_payload<RayPayload>();
    payload.uv = uv;
    payload.t_hit = optixGetRayTmax();
    payload.material_id = params.material_id;
    payload.normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
}

