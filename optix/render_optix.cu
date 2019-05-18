#include <math_constants.h>
#include <optix.h>
#include <optix_math.h>

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

// Global camera parameters
rtDeclareVariable(float3, cam_pos, , );
rtDeclareVariable(float3, cam_du, , );
rtDeclareVariable(float3, cam_dv, , );
rtDeclareVariable(float3, cam_dir_top_left, , );

rtDeclareVariable(rtObject, scene, , );

rtDeclareVariable(uint2, pixel, rtLaunchIndex, );
rtDeclareVariable(uint2, screen, rtLaunchDim, );

rtBuffer<uchar4, 2> framebuffer;
rtBuffer<float4, 2> accum_buffer;

// View params buffer:
// camera position, dir_du, dir_dv, dir_top_left
struct ViewParams {
	float4 cam_pos;
	float4 cam_du;
	float4 cam_dv;
	float4 cam_dir_top_left;
	uint32_t frame_id;
};
rtBuffer<ViewParams, 1> view_params;

rtBuffer<int3, 1> index_buffer;
rtBuffer<float3, 1> vertex_buffer;

struct RayPayload {
	// float3 color, float depth
	float4 color_dist;
	// float3 normal, float 1/0 if occlusion hit
	float4 normal_hit;
};

__device__ RayPayload make_ray_payload() {
	RayPayload p;
	p.color_dist = make_float4(0.f);
	p.normal_hit = make_float4(0.f);
	return p;
}

rtDeclareVariable(RayPayload, ray_payload, rtPayload, );

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

__device__ uint32_t pcg32_random(PCGRand &rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005ULL + 1;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ float pcg32_randomf(PCGRand &rng) {
	return ldexp((double)pcg32_random(rng), -32);
}

__device__ PCGRand get_rng(uint32_t frame_id) {
	uint32_t seed = (pixel.x + pixel.y * screen.x) * (frame_id + 1);

	PCGRand rng;
	rng.state = 0;
	pcg32_random(rng);
	rng.state += seed;
	pcg32_random(rng);
	return rng;
}

__device__ float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

__device__ void ortho_basis(float3 &v_x, float3 &v_y, const float3 &n) {
	v_y = make_float3(0.f, 0.f, 0.f);

	if (n.x < 0.6f && n.x > -0.6f) {
		v_y.x = 1.f;
	} else if (n.y < 0.6f && n.y > -0.6f) {
		v_y.y = 1.f;
	} else if (n.z < 0.6f && n.z > -0.6f) {
		v_y.z = 1.f;
	} else {
		v_y.x = 1.f;
	}
	v_x = normalize(cross(v_y, n));
	v_y = normalize(cross(n, v_x));
}

RT_PROGRAM void perspective_camera() {
	const int PRIMARY_RAY = 0;
	const int OCCLUSION_RAY = 1;
	const int MAX_PATH_DEPTH = 5;

	const ViewParams view = view_params[0];
	PCGRand rng = get_rng(view.frame_id);
	const float2 d = make_float2(pixel.x + pcg32_randomf(rng), pixel.y + pcg32_randomf(rng)) / make_float2(screen);
	const float3 ray_dir = normalize(d.x * make_float3(view.cam_du)
			+ d.y * make_float3(view.cam_dv) + make_float3(view.cam_dir_top_left));

	optix::Ray ray(make_float3(view.cam_pos), ray_dir, PRIMARY_RAY, 0.0);

	const float3 light_emission = make_float3(1.0);
	int bounce = 0;
	float3 illum = make_float3(0.0);
	float3 path_throughput = make_float3(1.0);
	do {
		RayPayload payload = make_ray_payload();
		rtTrace(scene, ray, payload, RT_VISIBILITY_ALL,
				RTrayflags(RT_RAY_FLAG_DISABLE_ANYHIT));

		if (payload.color_dist.w <= 0) {
			break;
		}

		const float3 hit_p = ray.origin + payload.color_dist.w * ray.direction;
		float3 v_x, v_y;
		float3 v_z = make_float3(payload.normal_hit);
		ortho_basis(v_x, v_y, v_z);

		const float3 bsdf = make_float3(payload.color_dist) * M_1_PI;

		// Direct light sampling.
		const float3 w_o = -ray.direction;
		const float3 light_dir = normalize(make_float3(-0.5, 0.8, 0.5));

		optix::Ray shadow_ray(hit_p, light_dir, OCCLUSION_RAY, 0.0001);
		RayPayload shadow_payload = make_ray_payload();
		rtTrace(scene, shadow_ray, shadow_payload, RT_VISIBILITY_ALL,
				RTrayflags(RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT | RT_RAY_FLAG_DISABLE_ANYHIT));
		if (shadow_payload.normal_hit.w == 0.f) {
			illum += path_throughput * bsdf * light_emission * abs(dot(light_dir, v_z));
		}

		// Sample the hemisphere
		// TODO: Cosine weighted hemisphere sampling
		const float theta = sqrt(pcg32_randomf(rng));
		const float phi = 2.0f * CUDART_PI_F * pcg32_randomf(rng);

		const float x = cos(phi) * theta;
		const float y = sin(phi) * theta;
		const float z = sqrt(1.0 - theta * theta);

		float3 w_i;
		w_i.x = x * v_x.x + y * v_y.x + z * v_z.x;
		w_i.y = x * v_x.y + y * v_y.y + z * v_z.y;
		w_i.z = x * v_x.z + y * v_y.z + z * v_z.z;
		w_i = normalize(w_i);

		// Update path throughput and continue the ray
		// TODO: This is just a hard-coded Lambertian BRDF,
		// using the object's normal color as its albedo
		float pdf = abs(dot(w_i, v_z)) * M_1_PI;
		if (pdf == 0.0) {
			break;
		}
		// Note: same as just multiplying my M_PI b/c the cancellation,
		// but left like this b/c I'll swap to Disney BRDF soon-ish
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;
		if (path_throughput.x == 0 && path_throughput.y == 0 && path_throughput.z == 0) {
			break;
		}

		// Update ray
		ray.origin = hit_p;
		ray.direction = w_i;
		ray.tmin = 0.0001;
		ray.tmax = 1e20f;

		++bounce;
	} while (bounce < MAX_PATH_DEPTH);

	const float4 accum_color = (make_float4(illum, 1.0) + view.frame_id * accum_buffer[pixel]) / (view.frame_id + 1);
	accum_buffer[pixel] = accum_color;

	framebuffer[pixel] = make_uchar4(linear_to_srgb(accum_color.x) * 255.f,
			linear_to_srgb(accum_color.y) * 255.f,
			linear_to_srgb(accum_color.z) * 255.f, 255);
}

rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

RT_PROGRAM void closest_hit() {
	const int3 indices = index_buffer[rtGetPrimitiveIndex()];
	const float3 v0 = vertex_buffer[indices.x];
	const float3 v1 = vertex_buffer[indices.y];
	const float3 v2 = vertex_buffer[indices.z];
	const float3 normal = normalize(cross(v1 - v0, v2 - v0));
	ray_payload.color_dist = make_float4(0.9, 0.9, 0.9, t_hit);
	ray_payload.normal_hit = make_float4(normal, 1.f);
}

RT_PROGRAM void occlusion_hit() {
	ray_payload.normal_hit.w = 1;
}

