#include <metal_common>
#include <metal_stdlib>
#include <simd/simd.h>
#include "shader_types.h"

using namespace metal;
using namespace raytracing;

struct Geometry {
    device packed_float3 *vertices [[id(0)]];
    device packed_uint3 *indices [[id(1)]];
    // Can you have null pointers if it doesn't have some attribute?
    // Seems like it should be possible to do in the argument buffer, just make
    // the stride big enough for the case that it does have all parameters
    // and leave them unset
};

struct Mesh {
    device uint32_t *geometries [[id(0)]];
};

kernel void raygen(uint2 tid [[thread_position_in_grid]],
                   texture2d<float, access::write> render_target [[texture(0)]],
                   constant ViewParams &view_params [[buffer(0)]],
                   instance_acceleration_structure scene [[buffer(1)]],
                   device Geometry *geometries [[buffer(2)]],
                   device Mesh *meshes [[buffer(3)]],
                   device MTLAccelerationStructureInstanceDescriptor *instances [[buffer(4)]],
                   device float4x4 *instance_inverse_transforms [[buffer(5)]])
{
    const float2 pixel = float2(tid);
    const float2 d = (pixel + 0.5f) / float2(view_params.fb_dims);

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
        // Find the instance we hit, the mesh associated with it and the geometry
        // within that mesh to find the intersected triangle
        device const MTLAccelerationStructureInstanceDescriptor &instance =
            instances[hit_result.instance_id];
        device const Mesh &mesh = meshes[instance.accelerationStructureIndex];
        device const Geometry &geom = geometries[mesh.geometries[hit_result.geometry_id]];

        const uint3 indices = geom.indices[hit_result.primitive_id];
        const float3 va = geom.vertices[indices.x];
        const float3 vb = geom.vertices[indices.y];
        const float3 vc = geom.vertices[indices.z];

        float4x4 normal_transform =
            transpose(instance_inverse_transforms[hit_result.instance_id]);

        // Transform the normal into world space
        float3 normal = normalize(cross(vb - va, vc - va));
        normal = normalize((normal_transform * float4(normal, 0.f)).xyz);

        const float3 bary_coords = float3(1.f - hit_result.triangle_barycentric_coord.x -
                                              hit_result.triangle_barycentric_coord.y,
                                          hit_result.triangle_barycentric_coord.x,
                                          hit_result.triangle_barycentric_coord.y);

        float3 color = (normal + 1.f) * 0.5f;
        render_target.write(float4(color, 1.f), tid);
    } else {
        render_target.write(float4(0.f), tid);
    }
}

