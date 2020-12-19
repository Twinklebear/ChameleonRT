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

// Not sure if there's a cleaner way to pass a buffer of texture handles,
// Metal didn't like device texture2d<float> *textures as a buffer parameter
struct Texture {
    texture2d<float> tex [[id(0)]];
};

kernel void raygen(uint2 tid [[thread_position_in_grid]],
                   texture2d<float, access::write> render_target [[texture(0)]],
                   constant ViewParams &view_params [[buffer(0)]],
                   instance_acceleration_structure scene [[buffer(1)]],
                   device Geometry *geometries [[buffer(2)]],
                   device MTLAccelerationStructureInstanceDescriptor *instances [[buffer(3)]],
                   device Instance *instance_data_buf [[buffer(4)]],
                   device DisneyMaterial *materials [[buffer(5)]],
                   device Texture *textures [[buffer(6)]],
                   device QuadLight *lights [[buffer(7)]])
{
    const float2 pixel = float2(tid);
    const float2 d = (pixel + 0.5f) / float2(view_params.fb_dims);

    constexpr sampler texture_sampler(address::repeat, filter::linear);

    ray ray;
    ray.origin = view_params.cam_pos.xyz;
    ray.direction = normalize(d.x * view_params.cam_du.xyz + d.y * view_params.cam_dv.xyz +
                              view_params.cam_dir_top_left.xyz);
    ray.min_distance = 0.f;
    ray.max_distance = INFINITY;

    intersector<instancing, triangle_data> traversal;
    typename intersector<instancing, triangle_data>::result_type hit_result;

    traversal.assume_geometry_type(geometry_type::triangle);
    traversal.force_opacity(forced_opacity::opaque);

    hit_result = traversal.intersect(ray, scene);

    if (hit_result.type != intersection_type::none) {
        // Find the instance we hit and look up the geometry we hit within
        // that instance to find the triangle
        device const MTLAccelerationStructureInstanceDescriptor &instance =
            instances[hit_result.instance_id];
        device const Instance &instance_data = instance_data_buf[hit_result.instance_id];
        device const Geometry &geom =
            geometries[instance_data.geometries[hit_result.geometry_id]];

        const uint3 idx = geom.indices[hit_result.primitive_id];
        const float3 va = geom.vertices[idx.x];
        const float3 vb = geom.vertices[idx.y];
        const float3 vc = geom.vertices[idx.z];

        float4x4 normal_transform = transpose(instance_data.inverse_transform);

        // Transform the normal into world space
        float3 normal = normalize(cross(vb - va, vc - va));
        normal = normalize((normal_transform * float4(normal, 0.f)).xyz);

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

        const uint32_t material_id = instance_data.material_ids[hit_result.geometry_id];
        float3 color = materials[material_id].base_color;
        const uint32_t mask = as_type<uint32_t>(color.x);
        if (IS_TEXTURED_PARAM(mask)) {
            const uint32_t tex_id = GET_TEXTURE_ID(mask);
            color = textures[tex_id].tex.sample(texture_sampler, uv).xyz;
        }

        // float3 color = (normal + 1.f) * 0.5f;
        render_target.write(float4(color, 1.f), tid);
    } else {
        render_target.write(float4(0.f), tid);
    }
}

