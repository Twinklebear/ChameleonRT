#include <embree3/rtcore.isph>

typedef unsigned int8 uint8_t;
typedef unsigned int uint32_t;

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

vec3 operator-(const vec3 &a, const vec3 &b) {
	return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3 operator+(const vec3 &a, const vec3 &b) {
	return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

vec3 operator*(const vec3 &a, const float s) {
	return make_vec3(a.x * s, a.y * s, a.z * s);
}

struct ViewParams {
	vec3 pos, dir_du, dir_dv, dir_top_left;
};

struct Scene {
	RTCScene scene;
	uniform RTCIntersectContext *uniform context;
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

export void generate_primary_rays(void *uniform _scene, uniform RTCRayHitNp *uniform ray_hit,
		void *uniform _tile, const void *uniform _view_params,
		const uniform uint32_t *uniform indices, const uniform float *uniform verts)
{
	Scene *uniform scene = (Scene *uniform)_scene;
	const ViewParams *uniform view_params = (const ViewParams *uniform)_view_params;
	Tile *uniform tile = (Tile *uniform)_tile;

	foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t ray = j * tile->width + i;
		const float px_x = (i + tile->x + 0.5f) / tile->fb_width;
		const float px_y = (j + tile->y + 0.5f) / tile->fb_height;

		ray_hit->ray.org_x[ray] = view_params->pos.x;
		ray_hit->ray.org_y[ray] = view_params->pos.y;
		ray_hit->ray.org_z[ray] = view_params->pos.z;

		ray_hit->ray.dir_x[ray] = view_params->dir_du.x * px_x
			+ view_params->dir_dv.x * px_y + view_params->dir_top_left.x;

		ray_hit->ray.dir_y[ray] = view_params->dir_du.y * px_x
			+ view_params->dir_dv.y * px_y + view_params->dir_top_left.y;

		ray_hit->ray.dir_z[ray] = view_params->dir_du.z * px_x
			+ view_params->dir_dv.z * px_y + view_params->dir_top_left.z;

		normalize(ray_hit->ray.dir_x + ray, ray_hit->ray.dir_y + ray,
				ray_hit->ray.dir_z + ray);
	}

	rtcIntersectNp(scene->scene, scene->context, ray_hit, tile->width * tile->height);

	foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t ray = j * tile->width + i;
		const uint32_t pixel = (j * tile->width + i) * 3;

		// TODO: Here we could now flip and batch by geomID
		if (ray_hit->hit.geomID[ray] != (uint32_t)-1) {
			const uniform uint32_t *tri = indices + ray_hit->hit.primID[ray] * 3;

			const vec3 *vv0 = (vec3*)(verts + tri[0] * 4);
			const vec3 *vv1 = (vec3*)(verts + tri[1] * 4);
			const vec3 *vv2 = (vec3*)(verts + tri[2] * 4);
			
			const vec3 v0 = *vv0;
			const vec3 v1 = *vv1;
			const vec3 v2 = *vv2;

			vec3 normal = normalized(cross(v1 - v0, v2 - v0));
			normal = (normal + make_vec3(1.f)) * 0.5f;

			tile->data[pixel] = normal.x;
			tile->data[pixel + 1] = normal.y;
			tile->data[pixel + 2] = normal.z;
		}
	}
}

// Convert the RGBF32 tile to sRGB and write it to the RGBA8 framebuffer
export void tile_to_uint8(void *uniform _tile, uniform uint8_t *uniform fb) {
	Tile *uniform tile = (Tile *uniform)_tile;
	foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
		const uint32_t tile_px = (j * tile->width + i) * 3;
		const uint32_t fb_px = ((j + tile->y) * tile->fb_width + i + tile->x) * 4;
		fb[fb_px] = linear_to_srgb(tile->data[tile_px]) * 255.f;
		fb[fb_px + 1] = linear_to_srgb(tile->data[tile_px + 1]) * 255.f;
		fb[fb_px + 2] = linear_to_srgb(tile->data[tile_px + 2]) * 255.f;
		fb[fb_px + 3] = 255;
	}
}

