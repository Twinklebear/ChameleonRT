#include "util.h"
#include "pcg_rng.h"
#include "disney_bsdf.h"
#include "lights.h"

struct LaunchParams {
	float4 cam_pos;
	float4 cam_du;
	float4 cam_dv;
	float4 cam_dir_top_left;

	uint32_t frame_id;

	uchar4 *framebuffer;
	float4 *accum_buffer;

	OptixTraversableHandle scene;
};

extern "C" {
	__constant__ LaunchParams launch_params;
}

// TODO: This can be made to match the host-side struct nicer since we
// won't need to worry about any layout/padding weirdness
struct MaterialParams {
	float4 basecolor_metallic;
	float4 spec_rough_spectint_aniso;
	float4 sheen_sheentint_clearc_ccgloss;
	float4 ior_spectrans;
};

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

__device__ float3 sample_direct_light(const DisneyMaterial &mat, const float3 &hit_p,
		const float3 &n, const float3 &v_x, const float3 &v_y, const float3 &w_o, PCGRand &rng)
{
	float3 illum = make_float3(0.f);

	QuadLight light;
	light.emission = make_float3(5.f);
	light.normal = normalize(make_float3(0.5, -0.8, -0.5));
	light.position = 10.f * -light.normal;
	// TODO: This would be input from the scene telling us how the light is placed
	// For now we don't care
	ortho_basis(light.v_x, light.v_y, light.normal);
	light.width = 5.f;
	light.height = 5.f;

	RayPayload shadow_payload = make_ray_payload();
	uint2 payload_ptr;
	pack_ptr(&shadow_payload, payload_ptr.x, payload_ptr.y);

	// Sample the light to compute an incident light ray to this point
	{
		float3 light_pos = sample_quad_light_position(light, make_float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		float3 light_dir = light_pos - hit_p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
		float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

		optixTrace(launch_params.scene, hit_p, light_dir, EPSILON, light_dist, 0,
				0xff, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				OCCLUSION_RAY, 0, OCCLUSION_RAY,
				payload_ptr.x, payload_ptr.y);

		if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_payload.normal_hit.w == 0.f) {
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

				shadow_payload.normal_hit = make_float4(0.f);

				optixTrace(launch_params.scene, hit_p, w_i, EPSILON, light_dist, 0,
						0xff, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
						OCCLUSION_RAY, 0, OCCLUSION_RAY,
						payload_ptr.x, payload_ptr.y);

				if (shadow_payload.normal_hit.w == 0.f) {
					illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
				}
			}
		}
	}
	return illum;
}

struct RayGenParams {
	MaterialParams *mat_params;
};

extern "C" __global__ void __raygen__perspective_camera() {
	const RayGenParams &params = *(const RayGenParams*)optixGetSbtDataPointer();

	const uint2 pixel = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
	const uint2 screen = make_uint2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
	const uint32_t pixel_idx = pixel.x + pixel.y * screen.x;

	PCGRand rng = get_rng((pixel.x + pixel.y * screen.x) * (launch_params.frame_id + 1));
	const float2 d = make_float2(pixel.x + pcg32_randomf(rng), pixel.y + pcg32_randomf(rng)) / make_float2(screen);
	float3 ray_dir = normalize(d.x * make_float3(launch_params.cam_du)
			+ d.y * make_float3(launch_params.cam_dv) + make_float3(launch_params.cam_dir_top_left));

	float3 ray_origin = make_float3(launch_params.cam_pos);

	DisneyMaterial mat;
	mat.base_color = make_float3(params.mat_params[0].basecolor_metallic);
	mat.metallic = params.mat_params[0].basecolor_metallic.w;
	mat.specular = params.mat_params[0].spec_rough_spectint_aniso.x;
	mat.roughness = params.mat_params[0].spec_rough_spectint_aniso.y;
	mat.specular_tint = params.mat_params[0].spec_rough_spectint_aniso.z;
	mat.anisotropy = params.mat_params[0].spec_rough_spectint_aniso.w;
	mat.sheen = params.mat_params[0].sheen_sheentint_clearc_ccgloss.x;
	mat.sheen_tint = params.mat_params[0].sheen_sheentint_clearc_ccgloss.y;
	mat.clearcoat = params.mat_params[0].sheen_sheentint_clearc_ccgloss.z;
	mat.clearcoat_gloss = params.mat_params[0].sheen_sheentint_clearc_ccgloss.w;
	mat.ior = params.mat_params[0].ior_spectrans.x;
	mat.specular_transmission = params.mat_params[0].ior_spectrans.y;

	const float3 light_emission = make_float3(1.0);
	int bounce = 0;
	float3 illum = make_float3(0.0);
	float3 path_throughput = make_float3(1.0);
	do {
		RayPayload payload = make_ray_payload();
		uint2 payload_ptr;
		pack_ptr(&payload, payload_ptr.x, payload_ptr.y);

		optixTrace(launch_params.scene, ray_origin, ray_dir, EPSILON, 1e20f, 0,
				0xff, OPTIX_RAY_FLAG_DISABLE_ANYHIT, PRIMARY_RAY, 0, PRIMARY_RAY,
				payload_ptr.x, payload_ptr.y);

		if (payload.color_dist.w <= 0) {
			illum = illum + path_throughput * make_float3(payload.color_dist);
			break;
		}

		const float3 w_o = -ray_dir;
		const float3 hit_p = ray_origin + payload.color_dist.w * ray_dir;
		float3 v_x, v_y;
		float3 v_z = make_float3(payload.normal_hit);
		ortho_basis(v_x, v_y, v_z);

		illum = illum + path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o, rng);

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

		ray_origin = hit_p;
		ray_dir = w_i;

		++bounce;
	} while (bounce < MAX_PATH_DEPTH);

	const float3 prev_color = make_float3(launch_params.accum_buffer[pixel_idx]);
	const float3 accum_color = (illum + launch_params.frame_id * prev_color) / (launch_params.frame_id + 1);
	launch_params.accum_buffer[pixel_idx] = make_float4(accum_color, 1.f);

	launch_params.framebuffer[pixel_idx] = make_uchar4(
			clamp(linear_to_srgb(accum_color.x) * 255.f, 0.f, 255.f),
			clamp(linear_to_srgb(accum_color.y) * 255.f, 0.f, 255.f),
			clamp(linear_to_srgb(accum_color.z) * 255.f, 0.f, 255.f), 255);
}

extern "C" __global__ void __miss__miss() {
	RayPayload *payload = get_payload<RayPayload>();
	payload->color_dist.w = -1;
	float3 dir = optixGetWorldRayDirection();
	// Apply our miss "shader" to draw the checkerboard background
	float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
	float v = acos(dir.y) * M_1_PI;

	int check_x = u * 10.f;
	int check_y = v * 10.f;

	if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
		payload->color_dist.x = 0.5f;
		payload->color_dist.y = 0.5f;
		payload->color_dist.z = 0.5f;
	} else {
		payload->color_dist.x = 0.1f;
		payload->color_dist.y = 0.1f;
		payload->color_dist.z = 0.1f;
	}
}

extern "C" __global__ void __miss__occlusion_miss() {
	RayPayload *payload = get_payload<RayPayload>();
	payload->color_dist.w = -1;
}

struct HitGroupParams {
	float3 *vertex_buffer;
	uint3 *index_buffer;
};

extern "C" __global__ void __closesthit__closest_hit() {
	const HitGroupParams &params = *(const HitGroupParams*)optixGetSbtDataPointer();

	// TODO: Barycentrics need to be queried via optixGetAttribute_0 & 1
	const uint3 indices = params.index_buffer[optixGetPrimitiveIndex()];
	const float3 v0 = params.vertex_buffer[indices.x];
	const float3 v1 = params.vertex_buffer[indices.y];
	const float3 v2 = params.vertex_buffer[indices.z];
	const float3 normal = normalize(cross(v1 - v0, v2 - v0));

	RayPayload *payload = get_payload<RayPayload>();
	payload->color_dist = make_float4(0.9, 0.9, 0.9, optixGetRayTmax());
	payload->normal_hit = make_float4(normal, 1.f);
}

extern "C" __global__ void __closesthit__occlusion_hit() {
	RayPayload *payload = get_payload<RayPayload>();
	payload->normal_hit.w = 1;
}

