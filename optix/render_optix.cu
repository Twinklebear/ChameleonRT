#include <optix.h>
#include <optix_math.h>

// Global camera parameters
rtDeclareVariable(float3, cam_pos, , );
rtDeclareVariable(float3, cam_du, , );
rtDeclareVariable(float3, cam_dv, , );
rtDeclareVariable(float3, cam_dir_top_left, , );

rtDeclareVariable(rtObject, model, , );

rtDeclareVariable(uint2, pixel, rtLaunchIndex, );

rtBuffer<uchar4, 2> framebuffer;

rtBuffer<int3, 1> index_buffer;
rtBuffer<float3, 1> vertex_buffer;

// Per-ray data
rtDeclareVariable(float3, prd_color, rtPayload, );

__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

RT_PROGRAM void perspective_camera() {
	optix::size_t2 screen = framebuffer.size();
	const float2 d = make_float2(pixel) / make_float2(screen);
	const float3 ray_dir = normalize(d.x * cam_du + d.y * cam_dv + cam_dir_top_left);

	optix::Ray ray(cam_pos, ray_dir, 0, 0.0001);

	float3 color = make_float3(0);
	rtTrace(model, ray, color, RT_VISIBILITY_ALL,
			RTrayflags(RT_RAY_FLAG_DISABLE_ANYHIT));
	framebuffer[pixel] = make_uchar4(linear_to_srgb(color.x) * 255.f,
			linear_to_srgb(color.y) * 255.f,
			linear_to_srgb(color.z) * 255.f, 255);
}

RT_PROGRAM void closest_hit() {
	const int3 indices = index_buffer[rtGetPrimitiveIndex()];
	const float3 v0 = vertex_buffer[indices.x];
	const float3 v1 = vertex_buffer[indices.y];
	const float3 v2 = vertex_buffer[indices.z];
	const float3 normal = normalize(cross(v1 - v0, v2 - v0));
	prd_color = (normal + make_float3(1.f)) * 0.5f;
}

