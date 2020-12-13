#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
using namespace raytracing;

kernel void raygen(uint2 tid [[thread_position_in_grid]],
                   texture2d<float, access::write> render_target [[texture(0)]],
                   instance_acceleration_structure scene [[buffer(0)]])
{
    const float2 pixel = float2(tid);
    const float2 dims = float2(1280, 720);
    // The example traces an orthographic view of a triangle (just rendering in NDC)
    const float2 origin = ((pixel + 0.5f) / dims) * 2.f - 1.f;

    ray ray;
    ray.origin = float3(origin.x, -origin.y, 1.f);
    ray.direction = float3(0.f, 0.f, -1.f);
    ray.min_distance = 0.f;
    ray.max_distance = INFINITY;

    intersector<instancing, triangle_data> traversal;
    typename intersector<instancing, triangle_data>::result_type hit_result;

    traversal.assume_geometry_type(geometry_type::triangle);
    traversal.force_opacity(forced_opacity::opaque);
    // We're just rendering a single triangle so any hit is fine
    traversal.accept_any_intersection(true);

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

