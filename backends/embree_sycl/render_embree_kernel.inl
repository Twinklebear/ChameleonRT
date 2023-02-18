#include <embree4/rtcore.h>
#include "../../util/texture_channel_mask.h"
#include "disney_bsdf.h"
#include "embree_utils.h"
#include "float3.h"
#include "lcg_rng.h"
#include "lights.h"
#include "mat4.h"
#include "texture2d.h"
#include "util.h"

namespace kernel {

float textured_scalar_param(const float x,
                            const float2 &uv,
                            const embree::ISPCTexture2D *textures)
{
    const uint32_t mask = intbits(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return texture_channel(&textures[tex_id], uv, channel);
    }
    return x;
}

void unpack_material(DisneyMaterial &mat,
                     const embree::MaterialParams *p,
                     const embree::ISPCTexture2D *textures,
                     const float2 uv)
{
    uint32_t mask = intbits(p->base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color = make_float3(texture(&textures[tex_id], uv));
    } else {
        mat.base_color = make_float3(p->base_color.x, p->base_color.y, p->base_color.z);
    }

    mat.metallic = textured_scalar_param(p->metallic, uv, textures);
    mat.specular = textured_scalar_param(p->specular, uv, textures);
    mat.roughness = textured_scalar_param(p->roughness, uv, textures);
    mat.specular_tint = textured_scalar_param(p->specular_tint, uv, textures);
    mat.anisotropy = textured_scalar_param(p->anisotropy, uv, textures);
    mat.sheen = textured_scalar_param(p->sheen, uv, textures);
    mat.sheen_tint = textured_scalar_param(p->sheen_tint, uv, textures);
    mat.clearcoat = textured_scalar_param(p->clearcoat, uv, textures);
    mat.clearcoat_gloss = textured_scalar_param(p->clearcoat_gloss, uv, textures);
    mat.ior = textured_scalar_param(p->ior, uv, textures);
    mat.specular_transmission = textured_scalar_param(p->specular_transmission, uv, textures);
}

float3 sample_direct_light(const embree::SceneContext &scene,
                           const DisneyMaterial &mat,
                           const float3 &hit_p,
                           const float3 &n,
                           const float3 &v_x,
                           const float3 &v_y,
                           const float3 &w_o,
                           QuadLight *lights,
                           uint32_t num_lights,
                           uint16_t &ray_stats,
                           LCGRand &rng)
{
    float3 illum = make_float3(0.f);

    uint32_t light_id = lcg_randomf(rng) * num_lights;
    light_id = sycl::min(light_id, num_lights - 1);
    QuadLight light = lights[light_id];

    RTCOccludedArguments occluded_args;
    rtcInitOccludedArguments(&occluded_args);
    occluded_args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
    occluded_args.feature_mask =
        (RTCFeatureFlags)(RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_INSTANCE);

    RTCRay shadow_ray;

    // TODO: move to use glm here on host and device?
    const float3 light_emission =
        make_float3(light.emission.x, light.emission.y, light.emission.z);

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos =
            sample_quad_light_position(light, make_float2(lcg_randomf(rng), lcg_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        set_ray(shadow_ray, hit_p, light_dir, EPSILON);
        shadow_ray.tfar = light_dist;
        rtcOccluded1(scene.scene, &shadow_ray, &occluded_args);
#ifdef REPORT_RAY_STATS
        ++ray_stats;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_ray.tfar > 0.f) {
            float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
            float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
            illum = bsdf * light_emission * sycl::fabs(dot(light_dir, n)) * w / light_pdf;
        }
    }

    // Sample the BRDF to compute a light sample as well
    {
        float3 w_i;
        float bsdf_pdf;
        float3 bsdf = sample_disney_brdf(mat, n, w_o, v_x, v_y, rng, w_i, bsdf_pdf);

        float light_dist;
        float3 light_pos;
        if (!all_zero(bsdf) && bsdf_pdf >= EPSILON &&
            quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
            float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
            if (light_pdf >= EPSILON) {
                float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
                set_ray(shadow_ray, hit_p, w_i, EPSILON);
                shadow_ray.tfar = light_dist;
                rtcOccluded1(scene.scene, &shadow_ray, &occluded_args);
#ifdef REPORT_RAY_STATS
                ++ray_stats;
#endif
                if (shadow_ray.tfar > 0.f) {
                    illum =
                        illum + bsdf * light_emission * sycl::fabs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

// A miss "shader" to make the same checkerboard background for testing as in the DXR backend
float3 miss_shader(const float3 &dir)
{
    float u = (1.f + sycl::atan2(dir.x, -dir.z) * M_1_PI_F) * 0.5f;
    float v = sycl::acos(dir.y) * M_1_PI_F;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1f && (check_x + check_y) % 2 == 0) {
        return make_float3(0.5f);
    }
    return make_float3(0.1f);
}

void trace_ray(const embree::SceneContext &scene,
               const embree::ViewParams &view_params,
               uint32_t i,
               uint32_t j)
{
    const uint32_t ray = i + scene.fb_width * j;

    uint16_t ray_stats = 0;
    float3 illum = make_float3(0.f);
    for (uint32_t s = 0; s < view_params.samples_per_pixel; ++s) {
        LCGRand rng = get_rng(i + scene.fb_width * j,
                              view_params.frame_id * view_params.samples_per_pixel + 1 + s);

        const float px_x = (i + lcg_randomf(rng)) / scene.fb_width;
        const float px_y = (j + lcg_randomf(rng)) / scene.fb_height;

        RTCRayHit path_ray;
        {
            float3 org = make_float3(view_params.pos.x, view_params.pos.y, view_params.pos.z);
            float3 dir = normalize(
                make_float3(view_params.dir_du.x * px_x + view_params.dir_dv.x * px_y +
                                view_params.dir_top_left.x,
                            view_params.dir_du.y * px_x + view_params.dir_dv.y * px_y +
                                view_params.dir_top_left.y,
                            view_params.dir_du.z * px_x + view_params.dir_dv.z * px_y +
                                view_params.dir_top_left.z));

            set_ray_hit(path_ray, org, dir, 0.f);
        }

        RTCIntersectArguments intersect_args;
        rtcInitIntersectArguments(&intersect_args);
        intersect_args.flags = RTC_RAY_QUERY_FLAG_COHERENT;
        intersect_args.feature_mask =
            (RTCFeatureFlags)(RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_INSTANCE);

        int bounce = 0;
        float3 path_throughput = make_float3(1.f);
        DisneyMaterial mat;
        mat4 matrix;
        do {
            rtcIntersect1(scene.scene, &path_ray, &intersect_args);
#ifdef REPORT_RAY_STATS
            ++ray_stats;
#endif
            intersect_args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

            const int inst = path_ray.hit.instID[0];
            const int geom = path_ray.hit.geomID;
            const int prim = path_ray.hit.primID;

            const float3 w_o =
                make_float3(-path_ray.ray.dir_x, -path_ray.ray.dir_y, -path_ray.ray.dir_z);

            if (prim == RTC_INVALID_GEOMETRY_ID) {
                illum = illum + path_throughput * miss_shader(neg(w_o));
                break;
            }

            const float3 hit_p =
                make_float3(path_ray.ray.org_x + path_ray.ray.tfar * path_ray.ray.dir_x,
                            path_ray.ray.org_y + path_ray.ray.tfar * path_ray.ray.dir_y,
                            path_ray.ray.org_z + path_ray.ray.tfar * path_ray.ray.dir_z);

            float3 v_z = normalize(
                make_float3(path_ray.hit.Ng_x, path_ray.hit.Ng_y, path_ray.hit.Ng_z));

            const float2 bary = make_float2(path_ray.hit.u, path_ray.hit.v);

            const embree::ISPCInstance *instance = nullptr;
            if (inst != RTC_INVALID_GEOMETRY_ID) {
                instance = &scene.instances[inst];
            } else {
                instance = &scene.instances[0];
            }
            const embree::ISPCGeometry *geometry = &instance->geometries[geom];

            float2 uv = make_float2(0.f, 0.f);
            const glm::uvec3 indices = geometry->index_buf[prim];

            if (geometry->uv_buf) {
                glm::vec2 uva = geometry->uv_buf[indices.x];
                glm::vec2 uvb = geometry->uv_buf[indices.y];
                glm::vec2 uvc = geometry->uv_buf[indices.z];
                glm::vec2 tmpuv = (1.f - bary.x - bary.y) * uva + bary.x * uvb + bary.y * uvc;
                uv.x = tmpuv.x;
                uv.y = tmpuv.y;
            }

            // Transform the normal back to world space
            if (instance) {
                // Transform the normal back to world space
                load_mat4(matrix, instance->world_to_object);
                transpose(matrix);
                v_z = normalize(mul(matrix, v_z));
            }

            unpack_material(
                mat, &scene.materials[instance->material_ids[geom]], scene.textures, uv);

            // Direct light sampling
            float3 v_x, v_y;
            if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.f) {
                v_z = neg(v_z);
            }
            ortho_basis(v_x, v_y, v_z);
            illum = illum + path_throughput * sample_direct_light(scene,
                                                                  mat,
                                                                  hit_p,
                                                                  v_z,
                                                                  v_x,
                                                                  v_y,
                                                                  w_o,
                                                                  scene.lights,
                                                                  scene.num_lights,
                                                                  ray_stats,
                                                                  rng);

            // Sample the BSDF to continue the ray
            float pdf;
            float3 w_i;
            float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
            if (pdf == 0.f || all_zero(bsdf)) {
                break;
            }
            path_throughput = path_throughput * bsdf * sycl::fabs(dot(w_i, v_z)) / pdf;

            // Trace the ray continuing the path
            set_ray_hit(path_ray, hit_p, w_i, EPSILON);
            ++bounce;

            // Russian roulette termination
            if (bounce > 3) {
                const float q = sycl::max(
                    0.05f,
                    1.f - sycl::max(path_throughput.x,
                                    sycl::max(path_throughput.y, path_throughput.z)));
                if (lcg_randomf(rng) < q) {
                    break;
                }
                path_throughput = path_throughput / (1.f - q);
            }
        } while (bounce < MAX_PATH_DEPTH);
    }
    illum = illum / view_params.samples_per_pixel;

#ifdef REPORT_RAY_STATS
    scene.ray_stats[ray] = ray_stats;
#endif

    const uint32_t px_id = ray * 3;

    const float3 accum = make_float3(scene.accum_buffer[px_id],
                                     scene.accum_buffer[px_id + 1],
                                     scene.accum_buffer[px_id + 2]);
    illum = (illum + view_params.frame_id * accum) / (view_params.frame_id + 1);

    scene.accum_buffer[px_id] = illum.x;
    scene.accum_buffer[px_id + 1] = illum.y;
    scene.accum_buffer[px_id + 2] = illum.z;

    scene.framebuffer[ray * 4] = glm::clamp(255.f * linear_to_srgb(illum.x), 0.f, 255.f);
    scene.framebuffer[ray * 4 + 1] = glm::clamp(255.f * linear_to_srgb(illum.y), 0.f, 255.f);
    scene.framebuffer[ray * 4 + 2] = glm::clamp(255.f * linear_to_srgb(illum.z), 0.f, 255.f);
    scene.framebuffer[ray * 4 + 3] = 255;
}

}
