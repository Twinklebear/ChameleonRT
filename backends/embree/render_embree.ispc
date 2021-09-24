#include <embree3/rtcore.isph>
#include "util.ih"
#include "lcg_rng.ih"
#include "float3.ih"
#include "mat4.ih"
#include "lights.ih"
#include "texture2d.ih"
#include "disney_bsdf.ih"
#include "util/texture_channel_mask.h"

struct ViewParams {
    float3 pos, dir_du, dir_dv, dir_top_left;
    uint32_t frame_id;
};

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
};

struct ISPCGeometry {
    const float4 *uniform vertex_buf;
    const uint3 *uniform index_buf;
    const float3 *uniform normal_buf;
    const float2 *uniform uv_buf;
};

struct ISPCInstance {
    const ISPCGeometry *uniform geometries;
    const float *uniform object_to_world;
    const float *uniform world_to_object;
    const uint32_t *uniform material_ids;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *uniform instances;
    MaterialParams *uniform materials;
    QuadLight *uniform lights;
    ISPCTexture2D *uniform textures;
    uniform uint32_t num_lights;
};

struct Tile {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t fb_width, fb_height;
    float *uniform data;
    uint16_t *uniform ray_stats;
};

float textured_scalar_param(const float x, const float2 &uv, const ISPCTexture2D *uniform textures) {
    const uint32_t mask = intbits(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return texture_channel(&textures[tex_id], uv, channel);
    }
    return x;
}

void unpack_material(DisneyMaterial &mat, const MaterialParams *p,
        const ISPCTexture2D *uniform textures, const float2 uv)
{
    uint32_t mask = intbits(p->base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color = make_float3(texture(&textures[tex_id], uv));
    } else {
        mat.base_color = p->base_color;
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

float3 sample_direct_light(const SceneContext *uniform scene,
        const DisneyMaterial &mat, const float3 &hit_p, const float3 &n,
        const float3 &v_x, const float3 &v_y, const float3 &w_o,
        RTCIntersectContext *uniform incoherent_context,
        QuadLight *uniform lights, uniform uint32_t num_lights,
        uint16_t &ray_stats, LCGRand &rng)
{
    float3 illum = make_float3(0.f);

    uint32_t light_id = lcg_randomf(rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    QuadLight light = lights[light_id];

    RTCRay shadow_ray;

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos = sample_quad_light_position(light, make_float2(lcg_randomf(rng), lcg_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        set_ray(shadow_ray, hit_p, light_dir, EPSILON);
        shadow_ray.tfar = light_dist;
        rtcOccludedV(scene->scene, incoherent_context, &shadow_ray);
#ifdef REPORT_RAY_STATS
        ++ray_stats;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_ray.tfar > 0.f) {
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
                set_ray(shadow_ray, hit_p, w_i, EPSILON);
                shadow_ray.tfar = light_dist;
                rtcOccludedV(scene->scene, incoherent_context, &shadow_ray);
#ifdef REPORT_RAY_STATS
                ++ray_stats;
#endif
                if (shadow_ray.tfar > 0.f) {
                    illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

// A miss "shader" to make the same checkerboard background for testing as in the DXR backend
float3 miss_shader(const float3 &dir) {
    float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
    float v = acos(dir.y) * M_1_PI;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1 && mod(check_x + check_y,  2) == 0) {
        return make_float3(0.5f);
    }
    return make_float3(0.1f);
}

export void trace_rays(void *uniform _scene, void *uniform _tile, const void *uniform _view_params)
{
    SceneContext *uniform scene = (SceneContext *uniform)_scene;
    const ViewParams *uniform view_params = (const ViewParams *uniform)_view_params;
    Tile *uniform tile = (Tile *uniform)_tile;
    uniform RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

    foreach (ray = 0 ... tile->width * tile->height)  {
        const uint32_t i = mod(ray, tile->width);
        const uint32_t j = ray / tile->width;

        LCGRand rng = get_rng((tile->x + i + (tile->y + j) * tile->fb_width), view_params->frame_id + 1);

        const float px_x = (i + tile->x + lcg_randomf(rng)) / tile->fb_width;
        const float px_y = (j + tile->y + lcg_randomf(rng)) / tile->fb_height;

        RTCRayHit path_ray;
        {
            float3 org = make_float3(view_params->pos.x, view_params->pos.y, view_params->pos.z);
            float3 dir = normalize(make_float3(
                        view_params->dir_du.x * px_x + view_params->dir_dv.x * px_y + view_params->dir_top_left.x,
                        view_params->dir_du.y * px_x + view_params->dir_dv.y * px_y + view_params->dir_top_left.y,
                        view_params->dir_du.z * px_x + view_params->dir_dv.z * px_y + view_params->dir_top_left.z));

            set_ray_hit(path_ray, org, dir, 0.f);
        }

        int bounce = 0;
        uint16_t ray_stats = 0;
        float3 illum = make_float3(0.0);
        float3 path_throughput = make_float3(1.0);
        DisneyMaterial mat;
        mat4 matrix;
        do {
            rtcIntersectV(scene->scene, &context, &path_ray);
#ifdef REPORT_RAY_STATS
            ++ray_stats;
#endif
            context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

            const int inst = path_ray.hit.instID[0];
            const int geom = path_ray.hit.geomID;
            const int prim = path_ray.hit.primID;

            const float3 w_o = make_float3(-path_ray.ray.dir_x, -path_ray.ray.dir_y, -path_ray.ray.dir_z);

            if (geom == RTC_INVALID_GEOMETRY_ID || inst == RTC_INVALID_GEOMETRY_ID
                    || prim == RTC_INVALID_GEOMETRY_ID)
            {
                illum = illum + path_throughput * miss_shader(neg(w_o));
                break;
            }

            const float3 hit_p = make_float3(path_ray.ray.org_x + path_ray.ray.tfar * path_ray.ray.dir_x,
                    path_ray.ray.org_y + path_ray.ray.tfar * path_ray.ray.dir_y,
                    path_ray.ray.org_z + path_ray.ray.tfar * path_ray.ray.dir_z);

            float3 normal = normalize(make_float3(path_ray.hit.Ng_x,
                        path_ray.hit.Ng_y,
                        path_ray.hit.Ng_z));

            const float2 bary = make_float2(path_ray.hit.u, path_ray.hit.v);

            const ISPCInstance *instance = &scene->instances[inst];
            const ISPCGeometry *geometry = &instance->geometries[geom];

            float2 uv = make_float2(0.f, 0.f);
            const uint3 indices = geometry->index_buf[prim];

            if (geometry->uv_buf) {
                float2 uva = geometry->uv_buf[indices.x];
                float2 uvb = geometry->uv_buf[indices.y];
                float2 uvc = geometry->uv_buf[indices.z];
                uv = (1.f - bary.x - bary.y) * uva
                    + bary.x * uvb + bary.y * uvc;
            }

            // Transform the normal back to world space
            load_mat4(matrix, instance->world_to_object);
            transpose(matrix);
            normal = normalize(mul(matrix, normal));

            unpack_material(mat, &scene->materials[instance->material_ids[geom]],
                    scene->textures, uv);

            // Direct light sampling
            float3 v_x, v_y;
            if (mat.specular_transmission == 0.f && dot(w_o, normal) < 0.0) {
                normal = neg(normal);
            }
            ortho_basis(v_x, v_y, normal);
            illum = illum + path_throughput
                * sample_direct_light(scene, mat, hit_p, normal, v_x, v_y, w_o, &context,
                        scene->lights, scene->num_lights, ray_stats, rng);

            // Sample the BSDF to continue the ray
            float pdf;
            float3 w_i;
            float3 bsdf = sample_disney_brdf(mat, normal, w_o, v_x, v_y, rng, w_i, pdf);
            if (pdf == 0.f || all_zero(bsdf)) {
                break;
            }
            path_throughput = path_throughput * bsdf * abs(dot(w_i, normal)) / pdf;

            // Trace the ray continuing the path
            set_ray_hit(path_ray, hit_p, w_i, EPSILON);
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

#ifdef REPORT_RAY_STATS
        tile->ray_stats[ray] = ray_stats;
#endif

        const uint32_t px_id = ray * 3;
        tile->data[px_id] = (illum.x + view_params->frame_id * tile->data[px_id]) / (view_params->frame_id + 1);
        tile->data[px_id + 1] = (illum.y + view_params->frame_id * tile->data[px_id + 1]) / (view_params->frame_id + 1);
        tile->data[px_id + 2] = (illum.z + view_params->frame_id * tile->data[px_id + 2]) / (view_params->frame_id + 1);
    }
}

// Convert the RGBF32 tile to sRGB and write it to the RGBA8 framebuffer
export void tile_to_uint8(void *uniform _tile, uniform uint8_t *uniform fb) {
    Tile *uniform tile = (Tile *uniform)_tile;
    foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
        const uint32_t tile_px = (j * tile->width + i) * 3;
        const uint32_t fb_px = ((j + tile->y) * tile->fb_width + i + tile->x) * 4;

        fb[fb_px] = float_to_srgb8(tile->data[tile_px]);
        fb[fb_px + 1] = float_to_srgb8(tile->data[tile_px + 1]);
        fb[fb_px + 2] = float_to_srgb8(tile->data[tile_px + 2]);
        fb[fb_px + 3] = 255;
    }
}

