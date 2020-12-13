#include <metal_stdlib>
#include <simd/simd.h>
#include "shader_types.h"

using namespace metal;
using namespace raytracing;

kernel void raygen(uint2 tid [[thread_position_in_grid]],
                   texture2d<float, access::write> render_target [[texture(0)]],
                   constant ViewParams &view_params [[buffer(0)]],
                   instance_acceleration_structure scene [[buffer(1)]])
{
    const float2 pixel = float2(tid);
    // The example traces an orthographic view of a triangle (just rendering in NDC)
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
        const float bary_a = 1.f - hit_result.triangle_barycentric_coord.x -
                             hit_result.triangle_barycentric_coord.y;
        render_target.write(float4(bary_a,
                                   hit_result.triangle_barycentric_coord.x,
                                   hit_result.triangle_barycentric_coord.y,
                                   1.f),
                            tid);
    } else {
        render_target.write(float4(0.f), tid);
    }
}

