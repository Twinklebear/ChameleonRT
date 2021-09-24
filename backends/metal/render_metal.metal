#include <metal_common>
#include <metal_stdlib>
#include <simd/simd.h>
#include "disney_bsdf.metal"
#include "lcg_rng.metal"
#include "lights.metal"
#include "shader_types.h"
#include "util.metal"
#include "util/texture_channel_mask.h"

using namespace metal;
using namespace raytracing;

struct Geometry {
    device packed_float3 *vertices [[id(0)]];
    device packed_uint3 *indices [[id(1)]];
    device packed_float3 *normals [[id(2)]];
    device packed_float2 *uvs [[id(3)]];
    uint32_t num_normals [[id(4)]];
    uint32_t num_uvs [[id(5)]];
};

struct Instance {
    float4x4 inverse_transform [[id(0)]];
    device uint32_t *geometries [[id(1)]];
    device uint32_t *material_ids [[id(2)]];
};

struct MaterialParams {
    packed_float3 base_color;
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

// Not sure if there's a cleaner way to pass a buffer of texture handles,
// Metal didn't like device texture2d<float> *textures as a buffer parameter
struct Texture {
    texture2d<float> tex [[id(0)]];
};

constexpr sampler texture_sampler(address::repeat, filter::linear);

float textured_scalar_param(const float x, const float2 uv, device const Texture *textures)
{
    const uint32_t mask = as_type<uint32_t>(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return textures[tex_id].tex.sample(texture_sampler, uv)[channel];
    }
    return x;
}

void unpack_material(thread DisneyMaterial &mat,
                     device const MaterialParams &p,
                     device const Texture *textures,
                     const float2 uv)
{
    uint32_t mask = as_type<uint32_t>(p.base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color = textures[tex_id].tex.sample(texture_sampler, uv).xyz;
    } else {
        mat.base_color = p.base_color;
    }

    mat.metallic = textured_scalar_param(p.metallic, uv, textures);
    mat.specular = textured_scalar_param(p.specular, uv, textures);
    mat.roughness = textured_scalar_param(p.roughness, uv, textures);
    mat.specular_tint = textured_scalar_param(p.specular_tint, uv, textures);
    mat.anisotropy = textured_scalar_param(p.anisotropy, uv, textures);
    mat.sheen = textured_scalar_param(p.sheen, uv, textures);
    mat.sheen_tint = textured_scalar_param(p.sheen_tint, uv, textures);
    mat.clearcoat = textured_scalar_param(p.clearcoat, uv, textures);
    mat.clearcoat_gloss = textured_scalar_param(p.clearcoat_gloss, uv, textures);
    mat.ior = textured_scalar_param(p.ior, uv, textures);
    mat.specular_transmission = textured_scalar_param(p.specular_transmission, uv, textures);
}

float3 sample_direct_light(instance_acceleration_structure scene,
                           thread const DisneyMaterial &mat,
                           thread const float3 &hit_p,
                           thread const float3 &n,
                           thread const float3 &v_x,
                           thread const float3 &v_y,
                           thread const float3 &w_o,
                           device const QuadLight *lights,
                           const uint32_t num_lights,
                           thread uint &ray_count,
                           thread LCGRand &rng)
{
    float3 illum = 0.f;

    uint32_t light_id = lcg_randomf(rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    const QuadLight light = lights[light_id];

    ray shadow_ray;
    shadow_ray.origin = hit_p;
    shadow_ray.min_distance = EPSILON;
    shadow_ray.max_distance = INFINITY;

    intersector<instancing, triangle_data> traversal;
    typename intersector<instancing, triangle_data>::result_type hit_result;
    traversal.assume_geometry_type(geometry_type::triangle);
    traversal.force_opacity(forced_opacity::opaque);

    // Sample the light to compute an incident light ray to this point
    {
        float3 light_pos =
            sample_quad_light_position(light, float2(lcg_randomf(rng), lcg_randomf(rng)));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        shadow_ray.direction = light_dir;
        shadow_ray.max_distance = light_dist;
        hit_result = traversal.intersect(shadow_ray, scene);

#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif
        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON &&
            hit_result.type == intersection_type::none) {
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
        if (any(bsdf > 0.f) && bsdf_pdf >= EPSILON &&
            quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
            float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
            if (light_pdf >= EPSILON) {
                float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);

                shadow_ray.direction = w_i;
                shadow_ray.max_distance = light_dist;
                hit_result = traversal.intersect(shadow_ray, scene);
#ifdef REPORT_RAY_STATS
                ++ray_count;
#endif
                if (hit_result.type == intersection_type::none) {
                    illum += bsdf * light.emission.rgb * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }
    return illum;
}

// A miss "shader" to make the same checkerboard background for testing as in the DXR backend
float3 miss_shader(thread const float3 &dir)
{
    float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI_F) * 0.5f;
    float v = acos(dir.y) * M_1_PI_F;

    int check_x = u * 10.f;
    int check_y = v * 10.f;

    if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
        return float3(0.5f);
    }
    return float3(0.1f);
}

kernel void raygen(uint2 tid [[thread_position_in_grid]],
                   texture2d<float, access::write> render_target [[texture(0)]],
                   texture2d<float, access::read_write> accum_buffer [[texture(1)]],
#ifdef REPORT_RAY_STATS
                   texture2d<uint, access::write> ray_stats [[texture(2)]],
#endif
                   constant ViewParams &view_params [[buffer(0)]],
                   instance_acceleration_structure scene [[buffer(1)]],
                   const device Geometry *geometries [[buffer(2)]],
                   const device MTLAccelerationStructureInstanceDescriptor *instances
                   [[buffer(3)]],
                   const device Instance *instance_data_buf [[buffer(4)]],
                   const device MaterialParams *materials [[buffer(5)]],
                   const device Texture *textures [[buffer(6)]],
                   const device QuadLight *lights [[buffer(7)]])
{
    const float2 pixel = float2(tid);
    LCGRand rng = get_rng(tid.x + tid.y * view_params.fb_dims.x, view_params.frame_id);
    const float2 d =
        (pixel + float2(lcg_randomf(rng), lcg_randomf(rng))) / float2(view_params.fb_dims);

    ray ray;
    ray.origin = view_params.cam_pos.xyz;
    ray.direction = normalize(d.x * view_params.cam_du.xyz + d.y * view_params.cam_dv.xyz +
                              view_params.cam_dir_top_left.xyz);
    ray.min_distance = 0.f;
    ray.max_distance = INFINITY;

    DisneyMaterial mat;
    uint ray_count = 0;
    int bounce = 0;
    float3 illum = float3(0, 0, 0);
    float3 path_throughput = float3(1, 1, 1);

    intersector<instancing, triangle_data> traversal;
    typename intersector<instancing, triangle_data>::result_type hit_result;
    traversal.assume_geometry_type(geometry_type::triangle);
    traversal.force_opacity(forced_opacity::opaque);

    do {
        hit_result = traversal.intersect(ray, scene);
#ifdef REPORT_RAY_STATS
        ++ray_count;
#endif

        if (hit_result.type == intersection_type::none) {
            illum += path_throughput * miss_shader(ray.direction);
            break;
        }

        // Find the instance we hit and look up the geometry we hit within
        // that instance to find the triangle data
        device const Instance &instance_data = instance_data_buf[hit_result.instance_id];
        device const Geometry &geom =
            geometries[instance_data.geometries[hit_result.geometry_id]];

        const uint3 idx = geom.indices[hit_result.primitive_id];
        const float3 va = geom.vertices[idx.x];
        const float3 vb = geom.vertices[idx.y];
        const float3 vc = geom.vertices[idx.z];

        const float3 bary = float3(hit_result.triangle_barycentric_coord.x,
                                   hit_result.triangle_barycentric_coord.y,
                                   1.f - hit_result.triangle_barycentric_coord.x -
                                       hit_result.triangle_barycentric_coord.y);

        float2 uv = float2(0.f);
        if (geom.num_uvs > 0) {
            const float2 uva = geom.uvs[idx.x];
            const float2 uvb = geom.uvs[idx.y];
            const float2 uvc = geom.uvs[idx.z];
            uv = bary.z * uva + bary.x * uvb + bary.y * uvc;
        }


        float3 normal = normalize(cross(vb - va, vc - va));
        /*
        if (geom.num_normals > 0) {
            const float3 na = geom.normals[idx.x];
            const float3 nb = geom.normals[idx.y];
            const float3 nc = geom.normals[idx.z];
            normal = normalize(bary.z * na + bary.x * nb + bary.y * nc);
        }
        */

        // Transform the normal into world space
        float4x4 normal_transform = transpose(instance_data.inverse_transform);
        normal = normalize((normal_transform * float4(normal, 0.f)).xyz);

        const uint32_t material_id = instance_data.material_ids[hit_result.geometry_id];

        const float3 w_o = -ray.direction;
        const float3 hit_p = ray.origin + hit_result.distance * ray.direction;

        unpack_material(mat, materials[material_id], textures, uv);

        float3 v_x, v_y;
        float3 v_z = normal;
        // For opaque objects (or in the future, thin ones) make the normal face forward
        if (mat.specular_transmission == 0.f && dot(w_o, v_z) < 0.0) {
            v_z = -v_z;
        }
        ortho_basis(v_x, v_y, v_z);

        illum += path_throughput * sample_direct_light(scene,
                                                       mat,
                                                       hit_p,
                                                       v_z,
                                                       v_x,
                                                       v_y,
                                                       w_o,
                                                       lights,
                                                       view_params.num_lights,
                                                       ray_count,
                                                       rng);

        float3 w_i;
        float pdf;
        float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
        if (pdf == 0.f || all(bsdf == 0.f)) {
            break;
        }
        path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

        ray.origin = hit_p;
        ray.direction = w_i;
        ray.min_distance = EPSILON;
        ray.max_distance = 1e20f;
        ++bounce;

        // Russian roulette termination
        if (bounce > 3) {
            const float q =
                max(0.05f,
                    1.f - max(path_throughput.x, max(path_throughput.y, path_throughput.z)));
            if (lcg_randomf(rng) < q) {
                break;
            }
            path_throughput = path_throughput / (1.f - q);
        }
    } while (bounce < MAX_PATH_DEPTH);

    const float3 accum_color = (illum + view_params.frame_id * accum_buffer.read(tid).xyz) /
                               (view_params.frame_id + 1);
    accum_buffer.write(float4(accum_color, 1.f), tid);
    render_target.write(float4(linear_to_srgb(accum_color.x),
                               linear_to_srgb(accum_color.y),
                               linear_to_srgb(accum_color.z),
                               1.f),
                        tid);
#ifdef REPORT_RAY_STATS
    ray_stats.write(ray_count, tid);
#endif
}

