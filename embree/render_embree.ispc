#include <embree3/rtcore_ray.isph>

typedef unsigned int8 uint8;
typedef unsigned int uint32;

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

void normalize(uniform float *x, uniform float *y, uniform float *z) {
	const float len = sqrt(*x * *x + *y * *y + *z * *z);
	*x /= len;
	*y /= len;
	*z /= len;
}

void normalize(float v[3]) {
	const float len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] /= len;
	v[1] /= len;
	v[2] /= len;
}

void cross(float a[3], float b[3], float out[3]) {
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

export void generate_primary_rays(uniform RTCRayHitNp *uniform ray_hit,
		const uniform uint32 tile_x, const uniform uint32 tile_y,
		const uniform uint32 fb_width, const uniform uint32 fb_height,
		const uniform uint32 tile_width, const uniform uint32 tile_height,
		const uniform float *uniform pos,
		const uniform float *uniform dir_du, const uniform float *uniform dir_dv,
		const uniform float *uniform dir_top_left)
{
	foreach (i = 0 ... tile_width, j = 0 ... tile_height) {
		const uint32 ray = j * tile_width + i;
		const float px_x = (i + tile_x + 0.5f) / fb_width;
		const float px_y = (j + tile_y + 0.5f) / fb_height;

		ray_hit->ray.org_x[ray] = pos[0];
		ray_hit->ray.org_y[ray] = pos[1];
		ray_hit->ray.org_z[ray] = pos[2];

		ray_hit->ray.dir_x[ray] = dir_du[0] * px_x + dir_dv[0] * px_y + dir_top_left[0];
		ray_hit->ray.dir_y[ray] = dir_du[1] * px_x + dir_dv[1] * px_y + dir_top_left[1];
		ray_hit->ray.dir_z[ray] = dir_du[2] * px_x + dir_dv[2] * px_y + dir_top_left[2];
		normalize(ray_hit->ray.dir_x + ray, ray_hit->ray.dir_y + ray,
				ray_hit->ray.dir_z + ray);
	}
}

export void shade_ray_stream(uniform RTCRayHitNp *uniform ray_hit,
		const uniform uint32 tile_width, const uniform uint32 tile_height,
		const uniform uint32 *uniform indices, const uniform float *uniform verts,
		uniform float *uniform tile)
{
	foreach (i = 0 ... tile_width, j = 0 ... tile_height) {
		const uint32 ray = j * tile_width + i;
		const uint32 pixel = (j * tile_width + i) * 3;

		// TODO: Here we could now flip and batch by geomID
		if (ray_hit->hit.geomID[ray] != (uint32)-1) {
			const uniform uint32 *tri = indices + ray_hit->hit.primID[ray] * 3;

			const uniform float *v0 = verts + tri[0] * 4;
			const uniform float *v1 = verts + tri[1] * 4;
			const uniform float *v2 = verts + tri[2] * 4;

			float normal[3] = {0.f};
			float a[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
			float b[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
			cross(a, b, normal);
			normalize(normal);

			normal[0] = (normal[0] + 1.f) * 0.5f;
			normal[1] = (normal[1] + 1.f) * 0.5f;
			normal[2] = (normal[2] + 1.f) * 0.5f;
			tile[pixel] = normal[0];
			tile[pixel + 1] = normal[1];
			tile[pixel + 2] = normal[2];
		}
	}
}

// Convert the RGBF32 tile to sRGB and write it to the RGBA8 framebuffer
export void tile_to_uint8(uniform float *uniform tile, uniform uint8 *uniform fb,
		const uniform uint32 fb_width, const uniform uint32 fb_height,
		const uniform uint32 tile_x, const uniform uint32 tile_y,
		const uniform uint32 tile_width, const uniform uint32 tile_height)
{
	foreach (i = 0 ... tile_width, j = 0 ... tile_height) {
		const uint32 tile_px = (j * tile_width + i) * 3;
		const uint32 fb_px = ((j + tile_y) * fb_width + i + tile_x) * 4;
		fb[fb_px] = linear_to_srgb(tile[tile_px]) * 255.f;
		fb[fb_px + 1] = linear_to_srgb(tile[tile_px + 1]) * 255.f;
		fb[fb_px + 2] = linear_to_srgb(tile[tile_px + 2]) * 255.f;
		fb[fb_px + 3] = 255;
	}
}

