#include <embree3/rtcore.isph>
#include "util.ih"
#include "pcg_rng.ih"
#include "float3.ih"
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

struct ISPCInstance {
	const float4 *uniform vertex_buf;
	const uint3 *uniform index_buf;
	const float3 *uniform normal_buf;
	const float2 *uniform uv_buf;
	const float *uniform transform;
	const float *uniform inv_transform;
	const uniform uint32_t material_id;
	const uniform uint32_t num_tris;
};

struct SceneContext {
	RTCScene scene;
	RTCIntersectContext *uniform coherent_context;
	RTCIntersectContext *uniform incoherent_context;
	ISPCInstance *uniform instances;
	MaterialParams *uniform material_params;
};

struct Tile {
	uint32_t x, y;
	uint32_t width, height;
	uint32_t fb_width, fb_height;
	float *uniform data;
};

void unpack_material(DisneyMaterial &mat, const MaterialParams *p, const float2 uv) {
	if (p->color_texture) {
		//mat.base_color = make_float3(uv.x, uv.y, 0.f);
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

float3 shade_path(const SceneContext *uniform scene,
		const RTCRayHitNp *uniform ray_hits, uint32_t px, PCGRand &rng);

export void trace_rays(void *uniform _scene, RTCRayHitNp *uniform ray_hit,
		void *uniform _tile, const void *uniform _view_params)
{
	SceneContext *uniform scene = (SceneContext *uniform)_scene;
	const ViewParams *uniform view_params = (const ViewParams *uniform)_view_params;
	Tile *uniform tile = (Tile *uniform)_tile;

	foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t ray = j * tile->width + i;
		PCGRand rng = get_rng((tile->x + i + (tile->y + j) * tile->fb_width) * (view_params->frame_id + 1));

		const float px_x = (i + tile->x + pcg32_randomf(rng)) / tile->fb_width;
		const float px_y = (j + tile->y + pcg32_randomf(rng)) / tile->fb_height;

		ray_hit->ray.org_x[ray] = view_params->pos.x;
		ray_hit->ray.org_y[ray] = view_params->pos.y;
		ray_hit->ray.org_z[ray] = view_params->pos.z;
		ray_hit->ray.tnear[ray] = 0.f;

		ray_hit->ray.dir_x[ray] = view_params->dir_du.x * px_x
			+ view_params->dir_dv.x * px_y + view_params->dir_top_left.x;

		ray_hit->ray.dir_y[ray] = view_params->dir_du.y * px_x
			+ view_params->dir_dv.y * px_y + view_params->dir_top_left.y;

		ray_hit->ray.dir_z[ray] = view_params->dir_du.z * px_x
			+ view_params->dir_dv.z * px_y + view_params->dir_top_left.z;

		const float len = sqrt(pow2(ray_hit->ray.dir_x[ray]) + pow2(ray_hit->ray.dir_y[ray])
				+ pow2(ray_hit->ray.dir_z[ray]));
		ray_hit->ray.dir_x[ray] /= len;
		ray_hit->ray.dir_y[ray] /= len;
		ray_hit->ray.dir_z[ray] /= len;

		ray_hit->ray.tfar[ray] = 1e20f;

		ray_hit->hit.Ng_x[ray] = 0.f;
		ray_hit->hit.Ng_y[ray] = 0.f;
		ray_hit->hit.Ng_z[ray] = 0.f;
		ray_hit->hit.u[ray] = 0.f;
		ray_hit->hit.v[ray] = 0.f;

		ray_hit->hit.primID[ray] = RTC_INVALID_GEOMETRY_ID;
		ray_hit->hit.geomID[ray] = RTC_INVALID_GEOMETRY_ID;
		ray_hit->hit.instID[0][ray] = RTC_INVALID_GEOMETRY_ID;
	}
	rtcIntersectNp(scene->scene, scene->coherent_context, ray_hit, tile->width * tile->height);

	// Essentially our "closest hit" shader here. Compute the normal for each hit
	// Since the AO (or path tracing rays) will now be pretty incoherent tracing
	// a huge stream may not really be worth it, so we switch to just doing packets of
	// SIMD-width rays
	foreach_tiled (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t ray = j * tile->width + i;
		PCGRand rng = get_rng((tile->x + i + (tile->y + j) * tile->fb_width) * (view_params->frame_id + 2));

		const float3 color = shade_path(scene, ray_hit, ray, rng);
		
		const uint32_t pixel = ray * 3;
		tile->data[pixel] = (color.x + view_params->frame_id * tile->data[pixel]) / (view_params->frame_id + 1);
		tile->data[pixel + 1] = (color.y + view_params->frame_id * tile->data[pixel + 1]) / (view_params->frame_id + 1);
		tile->data[pixel + 2] = (color.z + view_params->frame_id * tile->data[pixel + 2]) / (view_params->frame_id + 1);
	}
}

float3 sample_direct_light(const SceneContext *uniform scene,
		const DisneyMaterial &mat, const float3 &hit_p, const float3 &n,
		const float3 &v_x, const float3 &v_y, const float3 &w_o, PCGRand &rng)
{
	float3 illum = make_float3(0.f);

	QuadLight light;
	light.emission = make_float3(5.f);
	light.normal = normalize(make_float3(0.5, -0.8, -0.5));
	light.position = 10.f * neg(light.normal);
	// TODO: This would be input from the scene telling us how the light is placed
	// For now we don't care
	ortho_basis(light.v_x, light.v_y, light.normal);
	light.width = 5.f;
	light.height = 5.f;

	RTCRay shadow_ray;

	// Sample the light to compute an incident light ray to this point
	{
		float3 light_pos = sample_quad_light_position(light, make_float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		float3 light_dir = light_pos - hit_p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
		float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

		set_ray(shadow_ray, hit_p, light_dir, EPSILON);
		shadow_ray.tfar = light_dist;
		rtcOccludedV(scene->scene, scene->incoherent_context, &shadow_ray);

		if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_ray.tfar > 0.f) {
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

				set_ray(shadow_ray, hit_p, w_i, EPSILON);
				shadow_ray.tfar = light_dist;
				rtcOccludedV(scene->scene, scene->incoherent_context, &shadow_ray);
				if (shadow_ray.tfar > 0.f) {
					illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
				}
			}
		}
	}
	return illum;
}

// A miss "shader" to make the same checkerboard background for testing as in the DXR backend
float3 miss_shader(const float3 &dir) {
	float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
	float v = acos(dir.y) * M_1_PI;

	int check_x = u * 10.f;
	int check_y = v * 10.f;

	if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
		return make_float3(0.5f);// * (1.f + normalize(dir)) * 0.5f;
	}
	return make_float3(0.1f);
}

float3 shade_path(const SceneContext *uniform scene,
		const RTCRayHitNp *uniform ray_hits, uint32_t px, PCGRand &rng)
{
	int inst = ray_hits->hit.instID[0][px];
	int geom = ray_hits->hit.geomID[px];
	int prim = ray_hits->hit.primID[px];

	if (geom == RTC_INVALID_GEOMETRY_ID || inst == RTC_INVALID_GEOMETRY_ID || prim == RTC_INVALID_GEOMETRY_ID) {
		float3 dir = make_float3(ray_hits->ray.dir_x[px], ray_hits->ray.dir_y[px], ray_hits->ray.dir_z[px]);
		return miss_shader(dir);
	}

	// Initial hit data since we're coming from the stream of primary hits
	float3 hit_p;
	hit_p.x = ray_hits->ray.org_x[px]
		+ ray_hits->ray.tfar[px] * ray_hits->ray.dir_x[px];

	hit_p.y = ray_hits->ray.org_y[px]
		+ ray_hits->ray.tfar[px] * ray_hits->ray.dir_y[px];

	hit_p.z = ray_hits->ray.org_z[px]
		+ ray_hits->ray.tfar[px] * ray_hits->ray.dir_z[px];

	float3 normal;
	normal.x = ray_hits->hit.Ng_x[px];
	normal.y = ray_hits->hit.Ng_y[px];
	normal.z = ray_hits->hit.Ng_z[px];
	normal = normalize(normal);

	float3 w_o;
	w_o.x = -ray_hits->ray.dir_x[px];
	w_o.y = -ray_hits->ray.dir_y[px];
	w_o.z = -ray_hits->ray.dir_z[px];

	float2 bary;
	bary.x = ray_hits->hit.u[px];
	bary.y = ray_hits->hit.v[px];

	RTCRayHit path_ray;
	int bounce = 0;
	float3 illum = make_float3(0.0);
	float3 path_throughput = make_float3(1.0);
	float3 w_i;
	DisneyMaterial mat;
	do {
		const ISPCInstance *instance = &scene->instances[inst];

		float2 uv = make_float2(0.f, 0.f);
		if (prim >= instance->num_tris) {
			print("Prim % >= num tris % on bounce %!??\n", prim, instance->num_tris, bounce);
			break;
		}
		uint3 indices = instance->index_buf[prim];

		if (instance->uv_buf) {
			float2 uva = instance->uv_buf[indices.x];
			float2 uvb = instance->uv_buf[indices.y];
			float2 uvc = instance->uv_buf[indices.z];
			uv = (1.f - bary.x - bary.y) * uva
				+ bary.x * uvb + bary.y * uvc;
		}

		unpack_material(mat, &scene->material_params[instance->material_id], uv);

		// Direct light sampling
		float3 v_x, v_y;
		if (mat.specular_transmission == 0.f && dot(w_o, normal) < 0.0) {
			normal = neg(normal);
		}
		ortho_basis(v_x, v_y, normal);
		illum = illum + path_throughput
			* sample_direct_light(scene, mat, hit_p, normal, v_x, v_y, w_o, rng);

		// Sample the BSDF to continue the ray
		float pdf;
		float3 bsdf = sample_disney_brdf(mat, normal, w_o, v_x, v_y, rng, w_i, pdf);
		if (pdf < EPSILON || all_zero(bsdf)) {
			break;
		}
		path_throughput = path_throughput * bsdf * abs(dot(w_i, normal)) / pdf;

		if (path_throughput.x < EPSILON && path_throughput.y < EPSILON
				&& path_throughput.z < EPSILON)
		{
			break;
		}

		// Trace the ray continuing the path
		set_ray_hit(path_ray, hit_p, w_i, EPSILON);

		++bounce;
		if (bounce < MAX_PATH_DEPTH) {
			rtcIntersectV(scene->scene, scene->incoherent_context, &path_ray);

			inst = path_ray.hit.instID[0];
			geom = path_ray.hit.geomID;
			prim = path_ray.hit.primID;
			instance = 0;

			if (geom == RTC_INVALID_GEOMETRY_ID || inst == RTC_INVALID_GEOMETRY_ID
					|| prim == RTC_INVALID_GEOMETRY_ID)
			{
				illum = illum + path_throughput * miss_shader(w_i);
				break;
			}
			hit_p.x = path_ray.ray.org_x + path_ray.ray.tfar * w_i.x;
			hit_p.y = path_ray.ray.org_y + path_ray.ray.tfar * w_i.y;
			hit_p.z = path_ray.ray.org_z + path_ray.ray.tfar * w_i.z;

			normal.x = path_ray.hit.Ng_x;
			normal.y = path_ray.hit.Ng_y;
			normal.z = path_ray.hit.Ng_z;
			normal = normalize(normal);

			bary.x = path_ray.hit.u;
			bary.y = path_ray.hit.v;

			w_o = neg(w_i);
		}
	} while (bounce < MAX_PATH_DEPTH);

	return illum;
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

