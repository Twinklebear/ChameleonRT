#include "util.h"
#include "kernels/pcg_rng.h"
#include "kernels/disney_bsdf.h"
#include "kernels/lights.h"

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

struct MaterialParams {
	float4 basecolor_metallic;
	float4 spec_rough_spectint_aniso;
	float4 sheen_sheentint_clearc_ccgloss;
	float4 ior_spectrans;
};
rtBuffer<MaterialParams, 1> mat_params;

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

	optix::Ray shadow_ray(hit_p, make_float3(1.f), OCCLUSION_RAY, EPSILON);
	RayPayload shadow_payload = make_ray_payload();

	// Sample the light to compute an incident light ray to this point
	{
		float3 light_pos = sample_quad_light_position(light, make_float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		float3 light_dir = light_pos - hit_p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
		float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

		shadow_ray.direction = light_dir;
		shadow_ray.tmax = light_dist;
		rtTrace(scene, shadow_ray, shadow_payload, RT_VISIBILITY_ALL,
				RTrayflags(RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT | RT_RAY_FLAG_DISABLE_ANYHIT));

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
				shadow_ray.direction = w_i;
				shadow_ray.tmax = light_dist;
				rtTrace(scene, shadow_ray, shadow_payload, RT_VISIBILITY_ALL,
						RTrayflags(RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT | RT_RAY_FLAG_DISABLE_ANYHIT));
				if (shadow_payload.normal_hit.w == 0.f) {
					illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
				}
			}
		}
	}
	return illum;
}


RT_PROGRAM void perspective_camera() {
	const ViewParams view = view_params[0];

	PCGRand rng = get_rng((pixel.x + pixel.y * screen.x) * (view.frame_id + 1));
	const float2 d = make_float2(pixel.x + pcg32_randomf(rng), pixel.y + pcg32_randomf(rng)) / make_float2(screen);
	const float3 ray_dir = normalize(d.x * make_float3(view.cam_du)
			+ d.y * make_float3(view.cam_dv) + make_float3(view.cam_dir_top_left));

	optix::Ray ray(make_float3(view.cam_pos), ray_dir, PRIMARY_RAY, 0.0);

	DisneyMaterial mat;
	mat.base_color = make_float3(mat_params[0].basecolor_metallic);
	mat.metallic = mat_params[0].basecolor_metallic.w;
	mat.specular = mat_params[0].spec_rough_spectint_aniso.x;
	mat.roughness = mat_params[0].spec_rough_spectint_aniso.y;
	mat.specular_tint = mat_params[0].spec_rough_spectint_aniso.z;
	mat.anisotropy = mat_params[0].spec_rough_spectint_aniso.w;
	mat.sheen = mat_params[0].sheen_sheentint_clearc_ccgloss.x;
	mat.sheen_tint = mat_params[0].sheen_sheentint_clearc_ccgloss.y;
	mat.clearcoat = mat_params[0].sheen_sheentint_clearc_ccgloss.z;
	mat.clearcoat_gloss = mat_params[0].sheen_sheentint_clearc_ccgloss.w;
	mat.ior = mat_params[0].ior_spectrans.x;
	mat.specular_transmission = mat_params[0].ior_spectrans.y;

	const float3 light_emission = make_float3(1.0);
	int bounce = 0;
	float3 illum = make_float3(0.0);
	float3 path_throughput = make_float3(1.0);
	do {
		RayPayload payload = make_ray_payload();
		rtTrace(scene, ray, payload, RT_VISIBILITY_ALL,
				RTrayflags(RT_RAY_FLAG_DISABLE_ANYHIT));

		if (payload.color_dist.w <= 0) {
			float3 dir = ray.direction;
			// Apply our miss "shader" to draw the checkerboard background
			float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
			float v = acos(dir.y) * M_1_PI;

			int check_x = u * 10.f;
			int check_y = v * 10.f;

			float3 checker_color;
			if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
				checker_color = make_float3(0.5f);// * (1.f + normalize(dir)) * 0.5f;
			} else {
				checker_color = make_float3(0.1f);
			}
			illum += path_throughput * checker_color;
			break;
		}

		const float3 w_o = -ray.direction;
		const float3 hit_p = ray.origin + payload.color_dist.w * ray.direction;
		float3 v_x, v_y;
		float3 v_z = make_float3(payload.normal_hit);
		ortho_basis(v_x, v_y, v_z);

		illum += path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o, rng);

		float3 w_i;
		float pdf;
		float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
		if (pdf < EPSILON || all_zero(bsdf)) {
			break;
		}
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

		if (path_throughput.x < EPSILON && path_throughput.y < EPSILON && path_throughput.z < EPSILON) {
			break;
		}

		ray.origin = hit_p;
		ray.direction = w_i;
		ray.tmin = 0.0001;
		ray.tmax = 1e20f;

		++bounce;
	} while (bounce < MAX_PATH_DEPTH);

	const float4 accum_color = (make_float4(illum, 1.0) + view.frame_id * accum_buffer[pixel]) / (view.frame_id + 1);
	accum_buffer[pixel] = accum_color;

	framebuffer[pixel] = make_uchar4(clamp(linear_to_srgb(accum_color.x) * 255.f, 0.f, 255.f),
			clamp(linear_to_srgb(accum_color.y) * 255.f, 0.f, 255.f),
			clamp(linear_to_srgb(accum_color.z) * 255.f, 0.f, 255.f), 255);
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

