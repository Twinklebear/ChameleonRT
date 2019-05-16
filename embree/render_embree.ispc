#include <embree3/rtcore.isph>

#define M_PI 3.14159265358979323846

typedef unsigned int8 uint8_t;
typedef unsigned int uint32_t;
typedef unsigned int64 uint64_t;

struct vec3 {
	float x, y, z;
};

vec3 make_vec3(float x, float y, float z) {
	vec3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

vec3 make_vec3(float c) {
	return make_vec3(c, c, c);
}

float length(const vec3 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

vec3 normalized(const vec3 v) {
	float l = length(v);
	if (l < 0.0) {
		l = 0.0001;
	}
	const float c = 1.f / length(v);
	return make_vec3(v.x * c, v.y * c, v.z * c);
}

void normalize(uniform float *x, uniform float *y, uniform float *z) {
	const float len = sqrt(*x * *x + *y * *y + *z * *z);
	*x /= len;
	*y /= len;
	*z /= len;
}

vec3 cross(const vec3 a, const vec3 b) {
	vec3 c;
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

float dot(const vec3 a, const vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 operator-(const vec3 &a, const vec3 &b) {
	return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3 operator+(const vec3 &a, const vec3 &b) {
	return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

vec3 operator*(const vec3 &a, const float s) {
	return make_vec3(a.x * s, a.y * s, a.z * s);
}

void ortho_basis(vec3 &v_x, vec3 &v_y, const vec3 &n) {
	v_y = make_vec3(0, 0, 0);

	if (n.x < 0.6f && n.x > -0.6f) {
		v_y.x = 1.f;
	} else if (n.y < 0.6f && n.y > -0.6f) {
		v_y.y = 1.f;
	} else if (n.z < 0.6f && n.z > -0.6f) {
		v_y.z = 1.f;
	} else {
		v_y.x = 1.f;
	}
	v_x = normalized(cross(v_y, n));
	v_y = normalized(cross(n, v_x));
}

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

uint32_t pcg32_random(PCGRand &rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005ULL + 1;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float pcg32_randomf(PCGRand &rng) {
	return ldexp((double)pcg32_random(rng), -32);
}

PCGRand get_rng(uint32_t seed) {
	PCGRand rng;
	rng.state = 0;
	pcg32_random(rng);
	rng.state += seed;
	pcg32_random(rng);
	return rng;
}

struct ViewParams {
	vec3 pos, dir_du, dir_dv, dir_top_left;
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

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

RTCRay make_ray(const vec3 &pos, const vec3 &dir, const float tnear) {
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

RTCRayHit make_ray_hit(const vec3 &pos, const vec3 &dir, const float tnear) {
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

vec3 shade_hit(const Scene *uniform scene, const RTCRayHitNp *uniform ray_hits, varying uint32_t ray,
		const uint32_t *uniform indices, const float *uniform verts,
		varying PCGRand &rng);

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

		const vec3 color = shade_hit(scene, ray_hit, ray, indices, verts, rng);
		
		tile->data[pixel] = (color.x + view_params->frame_id * tile->data[pixel]) / (view_params->frame_id + 1);
		tile->data[pixel + 1] = (color.y + view_params->frame_id * tile->data[pixel + 1]) / (view_params->frame_id + 1);
		tile->data[pixel + 2] = (color.z + view_params->frame_id * tile->data[pixel + 2]) / (view_params->frame_id + 1);
	}
}

vec3 shade_hit(const Scene *uniform scene, const RTCRayHitNp *uniform ray_hits, varying uint32_t ray,
		const uint32_t *uniform indices, const float *uniform verts, varying PCGRand &rng)
{
	if (ray_hits->hit.geomID[ray] == (uint32_t)-1) {
		return make_vec3(0.f);
	}
	const uniform uint32_t *tri = indices + ray_hits->hit.primID[ray] * 3;

	const vec3 *vv0 = (vec3*)(verts + tri[0] * 4);
	const vec3 *vv1 = (vec3*)(verts + tri[1] * 4);
	const vec3 *vv2 = (vec3*)(verts + tri[2] * 4);

	const vec3 v0 = *vv0;
	const vec3 v1 = *vv1;
	const vec3 v2 = *vv2;

	vec3 normal;
	normal.x = ray_hits->hit.Ng_x[ray];
	normal.y = ray_hits->hit.Ng_y[ray];
	normal.z = ray_hits->hit.Ng_z[ray];
	normal = normalized(normal);

	vec3 v_x, v_y;
	ortho_basis(v_x, v_y, normal);

	vec3 hit_p;
	hit_p.x = ray_hits->ray.org_x[ray] + ray_hits->ray.tfar[ray] * ray_hits->ray.dir_x[ray];
	hit_p.y = ray_hits->ray.org_y[ray] + ray_hits->ray.tfar[ray] * ray_hits->ray.dir_y[ray];
	hit_p.z = ray_hits->ray.org_z[ray] + ray_hits->ray.tfar[ray] * ray_hits->ray.dir_z[ray];

	const float theta = sqrt(pcg32_randomf(rng));
	const float phi = 2.0f * M_PI * pcg32_randomf(rng);

	const float x = cos(phi) * theta;
	const float y = sin(phi) * theta;
	const float z = sqrt(1.0 - theta * theta);

	// TODO: Why do see some black spots (nans? but isnan says nothing?)
	// when letting it accumulate for a longer period?

	vec3 ao_dir;
	ao_dir.x = x * v_x.x + y * v_y.x + z * normal.x;
	ao_dir.y = x * v_x.y + y * v_y.y + z * normal.y;
	ao_dir.z = x * v_x.z + y * v_y.z + z * normal.z;

	RTCRay ao_ray = make_ray(hit_p, ao_dir, 0.0001);
	rtcOccludedV(scene->scene, scene->incoherent_context, &ao_ray);
	if (ao_ray.tfar < 0) {
		return make_vec3(0.f);
	}
	return (normal + make_vec3(1.f)) * 0.5f;
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

