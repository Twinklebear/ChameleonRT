#include <embree3/rtcore.isph>
#include "util.ih"
#include "pcg_rng.ih"
#include "float3.ih"
#include "mat4.ih"
#include "lights.ih"
#include "texture2d.ih"
#include "disney_bsdf.ih"

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

    ISPCTexture2D *uniform color_texture;
};

struct ISPCGeometry {
    const float4 *uniform vertex_buf;
    const uint3 *uniform index_buf;
    const float3 *uniform normal_buf;
    const float2 *uniform uv_buf;
    uniform uint32_t material_id;
};

struct ISPCInstance {
    const ISPCGeometry *uniform geometries;
    const float *uniform object_to_world;
    const float *uniform world_to_object;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *uniform instances;
    MaterialParams *uniform materials;
    QuadLight *uniform lights;
    uniform uint32_t num_lights;
};

struct RayGenParams {
    SceneContext *uniform scene;
    ViewParams *uniform view;
};

struct Tile {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t fb_width, fb_height;
    float *uniform data;
    uint16_t *uniform ray_stats;
};

struct ShaderTable {
    void *uniform raygen;

    uint8_t *uniform miss_shaders;
    uint32_t miss_stride;

    uint8_t *uniform hit_groups;
    uint32_t hit_group_stride;

    uint32_t *uniform instance_offset;
    ISPCInstance *uniform instances;
};

typedef void (*RayGen)(Tile *uniform tile, const uniform ShaderTable *uniform sbt,
        const int2 pixel, const void *uniform _params);

typedef void (*MissShader)(const RTCRay &ray, const void *params, void *uniform _payload);

typedef void (*ClosestHit)(const RTCRayHit &ray, const uniform float *varying object_to_world,
        const uniform float *varying world_to_object, const int prim_id,
        const void *params, void *uniform _payload);

export void execute_tile(const void *uniform _sbt, void *uniform _tile) {
    const uniform ShaderTable *uniform sbt = (const uniform ShaderTable *uniform)_sbt;
    Tile *uniform tile = (Tile *uniform)_tile;

    foreach (px = 0 ... tile->width * tile->height) {
        const uint32_t i = mod(px, tile->width);
        const uint32_t j = px / tile->width;
        // TODO: should just make the sbt a global
        uint64_t *uniform raygen_addr = (uint64_t *uniform)sbt->raygen;
        uniform RayGen raygen = (uniform RayGen)(*raygen_addr);
        void *uniform raygen_params = (void *uniform)(raygen_addr + 1);
        raygen(tile, sbt, make_int2(i, j), raygen_params);
    }
}

const int RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH = 1;
const int RAY_FLAG_SKIP_CLOSEST_HIT = 1 << 1;

void trace_ray(RTCScene scene, const float3 &ray_origin, const float3 &ray_dir, float tmin, float tmax,
        uint32_t ray_flags, uint32_t sbt_offset, uint32_t sbt_stride, uint32_t miss_index,
        uniform RTCIntersectContext *uniform context, const uniform ShaderTable *uniform sbt,
        void *uniform payload)
{
    RTCRayHit ray = make_ray_hit(ray_origin, ray_dir, tmin);
    ray.ray.tfar = tmax;
    bool missed = false;
    if (ray_flags & RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH) {
        rtcOccludedV(scene, context, &ray.ray);
        missed = ray.ray.tfar >= 0.f;
    } else {
        rtcIntersectV(scene, context, &ray);
        missed = ray.hit.geomID == RTC_INVALID_GEOMETRY_ID
            || ray.hit.instID[0] == RTC_INVALID_GEOMETRY_ID
            || ray.hit.primID == RTC_INVALID_GEOMETRY_ID;
    }

    const int inst = ray.hit.instID[0];
    const int geom = ray.hit.geomID;
    const int prim = ray.hit.primID;

    if (missed) {
        uint64_t *shader_addr = (uint64_t *)(sbt->miss_shaders + miss_index * sbt->miss_stride);
        void *params = (void *)(shader_addr + 1);
        varying MissShader shader = (varying MissShader)(*shader_addr);
        shader(ray.ray, params, payload);
    } else if (!(ray_flags & RAY_FLAG_SKIP_CLOSEST_HIT)) {
        const ISPCInstance *instance = &sbt->instances[inst];
        const int hg = sbt_offset + sbt_stride * geom + sbt->instance_offset[inst];
        uint64_t *shader_addr = (uint64_t *)(sbt->hit_groups + hg * sbt->hit_group_stride);
        void *params = (void *)(shader_addr + 1);
        varying ClosestHit shader = (varying ClosestHit)(*shader_addr);
        shader(ray, instance->object_to_world, instance->world_to_object,
                prim, params, payload);
    }
}

struct RayPayload {
    float2 uv;
    float t_hit;
    uint32_t material_id;

    float3 normal;
    float pad;
};

RayPayload make_ray_payload() {
    RayPayload p;
    p.uv = make_float2(0.f);
    p.t_hit = -1.f;
    p.material_id = 0;
    p.normal = make_float3(0.f);
    return p;
}

void unpack_material(DisneyMaterial &mat, const MaterialParams *p, const float2 uv) {
    if (p->color_texture) {
        mat.base_color = make_float3(texture(p->color_texture, uv));
    } else {
        mat.base_color = p->base_color;
    }
    mat.metallic = p->metallic;
    mat.specular = p->specular;
    mat.roughness = p->roughness;
    mat.specular_tint = p->specular_tint;
    mat.anisotropy = p->anisotropy;
    mat.sheen = p->sheen;
    mat.sheen_tint = p->sheen_tint;
    mat.clearcoat = p->clearcoat;
    mat.clearcoat_gloss = p->clearcoat_gloss;
    mat.ior = p->ior;
    mat.specular_transmission = p->specular_transmission;
}

float3 sample_direct_light(const SceneContext *uniform scene,
        const DisneyMaterial &mat, const float3 &hit_p, const float3 &n,
        const float3 &v_x, const float3 &v_y, const float3 &w_o,
        RTCIntersectContext *uniform context,
        QuadLight *uniform lights, uniform uint32_t num_lights,
        uint16_t &ray_stats, PCGRand &rng,
        const uniform ShaderTable *uniform sbt)
{
    float3 illum = make_float3(0.f);

    uint32_t light_id = pcg32_randomf(rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    QuadLight light = lights[light_id];

    int shadow_hit = 1;
    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos = sample_quad_light_position(light, make_float2(pcg32_randomf(rng), pcg32_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        trace_ray(scene->scene, hit_p, light_dir, EPSILON, light_dist,
                RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT,
                0, 0, 1,
                context, sbt, &shadow_hit);
#ifdef REPORT_RAY_STATS
        ++ray_stats;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && !shadow_hit) {
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

                shadow_hit = 1;
                trace_ray(scene->scene, hit_p, w_i, EPSILON, light_dist,
                        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT,
                        0, 0, 1,
                        context, sbt, &shadow_hit);
#ifdef REPORT_RAY_STATS
                ++ray_stats;
#endif
                if (!shadow_hit) {
                    illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

void raygen_perspective_camera(Tile *uniform tile, const uniform ShaderTable *uniform sbt,
        const int2 pixel, const void *uniform _params)
{
    const uniform RayGenParams *uniform params = (const uniform RayGenParams *uniform)_params;
    const uniform SceneContext *uniform scene = params->scene;
    const uniform ViewParams *uniform view_params = params->view;

    uniform RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

    PCGRand rng = get_rng((tile->x + pixel.x + (tile->y + pixel.y) * tile->fb_width) * (view_params->frame_id + 1));

    const float px_x = (pixel.x + tile->x + pcg32_randomf(rng)) / tile->fb_width;
    const float px_y = (pixel.y + tile->y + pcg32_randomf(rng)) / tile->fb_height;

    float3 ray_org = make_float3(view_params->pos.x, view_params->pos.y, view_params->pos.z);
    float3 ray_dir = normalize(make_float3(
                view_params->dir_du.x * px_x + view_params->dir_dv.x * px_y + view_params->dir_top_left.x,
                view_params->dir_du.y * px_x + view_params->dir_dv.y * px_y + view_params->dir_top_left.y,
                view_params->dir_du.z * px_x + view_params->dir_dv.z * px_y + view_params->dir_top_left.z));

    int bounce = 0;
    uint16_t ray_stats = 0;
    float3 illum = make_float3(0.f);
    float3 path_throughput = make_float3(1.0);
    DisneyMaterial mat;
    mat4 matrix;
    float tmin = 0.f;
    do {
        RayPayload payload = make_ray_payload();
        trace_ray(scene->scene, ray_org, ray_dir, tmin, 1e20f, 0, 0, 1, 0,
            &context, sbt, &payload);
        tmin = EPSILON;

#ifdef REPORT_RAY_STATS
        ++ray_stats;
#endif
        context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

        if (payload.t_hit <= 0.f) {
            illum = illum + path_throughput * payload.normal;
            break;
        }

        unpack_material(mat, &scene->materials[payload.material_id], payload.uv);

        const float3 w_o = neg(ray_dir);
        const float3 hit_p = ray_org + payload.t_hit * ray_dir;
        float3 v_x, v_y;
        float3 v_z = payload.normal;
        if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.0) {
            v_z = neg(v_z);
        }
        ortho_basis(v_x, v_y, v_z);

        illum = illum + path_throughput
            * sample_direct_light(scene, mat, hit_p, v_z, v_x, v_y, w_o, &context,
                    scene->lights, scene->num_lights, ray_stats, rng, sbt);

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

        ray_org = hit_p;
        ray_dir = w_i;

        ++bounce;
    } while (bounce < MAX_PATH_DEPTH);

    uint32_t px = pixel.y * tile->width + pixel.x;
#ifdef REPORT_RAY_STATS
    tile->ray_stats[px] = ray_stats;
#endif

    px = px * 3;
    tile->data[px] = (illum.x + view_params->frame_id * tile->data[px]) / (view_params->frame_id + 1);
    tile->data[px + 1] = (illum.y + view_params->frame_id * tile->data[px + 1]) / (view_params->frame_id + 1);
    tile->data[px + 2] = (illum.z + view_params->frame_id * tile->data[px + 2]) / (view_params->frame_id + 1);
}

// A miss "shader" to make the same checkerboard background for testing as in the DXR backend
void miss_shader(const RTCRay &ray, const void *params, void *uniform _payload) {
    const float3 dir = make_float3(ray.dir_x, ray.dir_y, ray.dir_z);
    varying RayPayload *uniform payload = (varying RayPayload *uniform)_payload;
    float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
    float v = acos(dir.y) * M_1_PI;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1 && mod(check_x + check_y,  2) == 0) {
        payload->normal = make_float3(0.5f);
    } else {
        payload->normal = make_float3(0.1f);
    }
}

void occlusion_miss_shader(const RTCRay &ray, const void *params, void *uniform _payload) {
    varying int *uniform payload = (varying int *uniform)_payload;
    *payload = 0;
}

void closest_hit(const RTCRayHit &ray, const uniform float *varying object_to_world,
        const uniform float *varying world_to_object,
        const int prim_id, const void *params, void *uniform _payload)
{
    varying RayPayload *uniform payload = (varying RayPayload *uniform)_payload;

    payload->normal = normalize(make_float3(ray.hit.Ng_x, ray.hit.Ng_y, ray.hit.Ng_z));

    const float2 bary = make_float2(ray.hit.u, ray.hit.v);

    // TODO: The transform mats should be passed as a param
    //const ISPCInstance *instance = &scene->instances[inst];
    const ISPCGeometry *geometry = (const ISPCGeometry *)params;
    const uint3 indices = geometry->index_buf[prim_id];

    payload->uv = make_float2(0.f, 0.f);
    if (geometry->uv_buf) {
        float2 uva = geometry->uv_buf[indices.x];
        float2 uvb = geometry->uv_buf[indices.y];
        float2 uvc = geometry->uv_buf[indices.z];
        payload->uv = (1.f - bary.x - bary.y) * uva
            + bary.x * uvb + bary.y * uvc;
    }

    payload->material_id = geometry->material_id;
    payload->t_hit = ray.ray.tfar;

    // Transform the normal back to world space
    mat4 matrix;
    load_mat4(matrix, world_to_object);
    transpose(matrix);
    payload->normal = normalize(mul(matrix, payload->normal));
}

export uniform uint64_t get_raygen_handle() {
    void *uniform x = raygen_perspective_camera;
    return (uniform uint64_t)x;
}

export uniform uint64_t get_miss_shader_handle() {
    void *uniform x = miss_shader;
    return (uniform uint64_t)x;
}

export uniform uint64_t get_occlusion_miss_shader_handle() {
    void *uniform x = occlusion_miss_shader;
    return (uniform uint64_t)x;
}

export uniform uint64_t get_closest_hit_handle() {
    void *uniform x = closest_hit;
    return (uniform uint64_t)x;
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


