#include <embree3/rtcore.isph>
#include "util.ih"
#include "pcg_rng.ih"
#include "float3.ih"

struct ViewParams {
	float3 pos, dir_du, dir_dv, dir_top_left;
	uint32_t frame_id;
};

struct Scene {
	RTCScene scene;
	uniform RTCIntersectContext *uniform coherent_context;
	uniform RTCIntersectContext *uniform incoherent_context;
};

struct Tile {
	uint32_t x, y;
	uint32_t width, height;
	uint32_t fb_width, fb_height;
	float *uniform data;
};

RTCRay make_ray(const float3 &pos, const float3 &dir, const float tnear) {
	RTCRay ray;
	ray.org_x = pos.x;
	ray.org_y = pos.y;
	ray.org_z = pos.z;
	ray.tnear = tnear;

	ray.dir_x = dir.x;
	ray.dir_y = dir.y;
	ray.dir_z = dir.z;
	ray.time = 0.f;
	ray.tfar = 1e20f;

	ray.mask = -1;
	ray.id = 0;
	ray.flags = 0;
	return ray;
}

RTCRayHit make_ray_hit(const float3 &pos, const float3 &dir, const float tnear) {
	RTCRayHit ray_hit;
	ray_hit.ray.org_x = pos.x;
	ray_hit.ray.org_y = pos.y;
	ray_hit.ray.org_z = pos.z;
	ray_hit.ray.tnear = tnear;

	ray_hit.ray.dir_x = dir.x;
	ray_hit.ray.dir_y = dir.y;
	ray_hit.ray.dir_z = dir.z;
	ray_hit.ray.time = 0.f;
	ray_hit.ray.tfar = 1e20f;

	ray_hit.ray.mask = -1;
	ray_hit.ray.id = 0;
	ray_hit.ray.flags = 0;

	ray_hit.hit.primID = -1;
	ray_hit.hit.geomID = -1;
	ray_hit.hit.instID[0] = -1;
	return ray_hit;
};

float3 shade_path(const Scene *uniform scene, const RTCRayHitNp *uniform ray_hits, uint32_t px, PCGRand &rng);

export void trace_rays(void *uniform _scene, RTCRayHitNp *uniform ray_hit,
		void *uniform _tile, const void *uniform _view_params,
		const uint32_t *uniform indices, const float *uniform verts)
{
	Scene *uniform scene = (Scene *uniform)_scene;
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

		ray_hit->ray.tfar[ray] = 1e20f;

		ray_hit->hit.Ng_x[ray] = 0.f;
		ray_hit->hit.Ng_y[ray] = 0.f;
		ray_hit->hit.Ng_z[ray] = 0.f;
		ray_hit->hit.u[ray] = 0.f;
		ray_hit->hit.v[ray] = 0.f;

		ray_hit->hit.primID[ray] = -1;
		ray_hit->hit.geomID[ray] = -1;
		ray_hit->hit.instID[0][ray] = -1;

		normalize(ray_hit->ray.dir_x + ray, ray_hit->ray.dir_y + ray,
				ray_hit->ray.dir_z + ray);
	}

	rtcIntersectNp(scene->scene, scene->coherent_context, ray_hit, tile->width * tile->height);

	// Essentially our "closest hit" shader here. Compute the normal for each hit
	// Since the AO (or path tracing rays) will now be pretty incoherent tracing
	// a huge stream may not really be worth it, so we switch to just doing packets of
	// SIMD-width rays
	foreach_tiled (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t ray = j * tile->width + i;
		const uint32_t pixel = ray * 3;
		PCGRand rng = get_rng((tile->x + i + (tile->y + j) * tile->fb_width) * (view_params->frame_id + 2));

		const float3 color = shade_path(scene, ray_hit, ray, rng);
		
		tile->data[pixel] = (color.x + view_params->frame_id * tile->data[pixel]) / (view_params->frame_id + 1);
		tile->data[pixel + 1] = (color.y + view_params->frame_id * tile->data[pixel + 1]) / (view_params->frame_id + 1);
		tile->data[pixel + 2] = (color.z + view_params->frame_id * tile->data[pixel + 2]) / (view_params->frame_id + 1);
	}
}

float3 shade_path(const Scene *uniform scene, const RTCRayHitNp *uniform ray_hits, uint32_t px, PCGRand &rng) {
	if (ray_hits->hit.geomID[px] == (uint32_t)-1) {
		return make_float3(0.f);
	}
	const int MAX_PATH_DEPTH = 5;

	// Initial hit data since we're coming from the stream of primary hits
	float3 hit_p;
	hit_p.x = ray_hits->ray.org_x[px] + ray_hits->ray.tfar[px] * ray_hits->ray.dir_x[px];
	hit_p.y = ray_hits->ray.org_y[px] + ray_hits->ray.tfar[px] * ray_hits->ray.dir_y[px];
	hit_p.z = ray_hits->ray.org_z[px] + ray_hits->ray.tfar[px] * ray_hits->ray.dir_z[px];

	float3 normal;
	normal.x = ray_hits->hit.Ng_x[px];
	normal.y = ray_hits->hit.Ng_y[px];
	normal.z = ray_hits->hit.Ng_z[px];
	normal = normalized(normal);

	float3 w_o;
	w_o.x = -ray_hits->ray.dir_x[px];
	w_o.y = -ray_hits->ray.dir_y[px];
	w_o.z = -ray_hits->ray.dir_z[px];

	const float3 light_emission = make_float3(1.0);
	const float3 light_dir = normalized(make_float3(-0.5, 0.8, 0.5));

	RTCRayHit path_ray;
	int bounce = 0;
	float3 illum = make_float3(0.0);
	float3 path_throughput = make_float3(1.0);
	float3 w_i;
	do {
		const float3 obj_color = make_float3(0.9);
		const float3 bsdf = obj_color * M_1_PI;

		// Direct light sampling
		RTCRay shadow_ray = make_ray(hit_p, light_dir, 0.0001);
		rtcOccludedV(scene->scene, scene->incoherent_context, &shadow_ray);
		if (shadow_ray.tfar > 0) {
			illum = illum + path_throughput * bsdf * light_emission * abs(dot(light_dir, normal));
		}

		// Sample the hemisphere
		// TODO: Cosine weighted hemisphere sampling
		float3 v_x, v_y;
		ortho_basis(v_x, v_y, normal);

		const float theta = sqrt(pcg32_randomf(rng));
		const float phi = 2.0f * M_PI * pcg32_randomf(rng);

		const float x = cos(phi) * theta;
		const float y = sin(phi) * theta;
		const float z = sqrt(1.0 - theta * theta);

		w_i.x = x * v_x.x + y * v_y.x + z * normal.x;
		w_i.y = x * v_x.y + y * v_y.y + z * normal.y;
		w_i.z = x * v_x.z + y * v_y.z + z * normal.z;
		w_i = normalized(w_i);

		float pdf = abs(dot(w_i, normal)) * M_1_PI;
		if (pdf == 0.0) {
			break;
		}
		// Note: same as just multiplying my M_PI b/c the cancellation,
		// but left like this b/c I'll swap to Disney BRDF soon-ish
		path_throughput = path_throughput * bsdf * abs(dot(w_i, normal)) / pdf;
		if (path_throughput.x == 0 && path_throughput.y == 0 && path_throughput.z == 0) {
			break;
		}

		path_ray = make_ray_hit(hit_p, w_i, 0.0001);

		++bounce;
		if (bounce < MAX_PATH_DEPTH) {
			rtcIntersectV(scene->scene, scene->incoherent_context, &path_ray);
			if (path_ray.hit.geomID == (uint32_t)-1) {
				break;
			}
			hit_p.x = path_ray.ray.org_x + path_ray.ray.tfar * w_i.x;
			hit_p.y = path_ray.ray.org_y + path_ray.ray.tfar * w_i.y;
			hit_p.z = path_ray.ray.org_z + path_ray.ray.tfar * w_i.z;

			normal.x = path_ray.hit.Ng_x;
			normal.y = path_ray.hit.Ng_y;
			normal.z = path_ray.hit.Ng_z;
			normal = normalized(normal);

			w_o.x = -w_i.x;
			w_o.y = -w_i.y;
			w_o.z = -w_i.z;
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

