#include <limits>
#include <iostream>
#include <tbb/parallel_for.h>
#include "render_embree.h"
#include "render_embree_ispc.h"

static float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

RenderEmbree::RenderEmbree()
	: device(rtcNewDevice(NULL)), scene(rtcNewScene(device))
{}

void RenderEmbree::initialize(const int fb_width, const int fb_height) {
	fb_dims = glm::ivec2(fb_width, fb_height);
	img.resize(fb_width * fb_height);
}

void RenderEmbree::set_mesh(const std::vector<float> &verts_unaligned,
		const std::vector<uint32_t> &idx)
{
	const size_t nverts = verts_unaligned.size() / 3;

	for (size_t i = 0; i < idx.size() / 3; ++i) {
		indices.push_back(glm::uvec3(idx[i * 3], idx[i * 3 + 1], idx[i * 3 + 2]));
	}

	// Pad the vertex buffer to be 16 bytes
	verts.reserve(nverts);
	for (size_t i = 0; i < nverts; ++i) {
		verts.push_back(glm::vec4(verts_unaligned[i * 3],
					verts_unaligned[i * 3 + 1],
					verts_unaligned[i * 3 + 2], 0.f));
	}

	RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	RTCBuffer vert_buf = rtcNewSharedBuffer(device, verts.data(),
			nverts * sizeof(glm::vec4));
	RTCBuffer index_buf = rtcNewSharedBuffer(device, indices.data(),
			indices.size() * sizeof(glm::uvec3));

	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
			vert_buf, 0, 16, verts.size());

	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
			index_buf, 0, 12, indices.size());

	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0);
	rtcCommitGeometry(geom);

	rtcAttachGeometry(scene, geom);
	rtcCommitScene(scene);
}

void RenderEmbree::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(fb_dims.x) / fb_dims.y;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	uint8_t *color = reinterpret_cast<uint8_t*>(img.data());

	const glm::uvec2 tile_size(64);
	// Round up the number of tiles we need to run in case the
	// framebuffer is not an even multiple of tile size
	const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
			fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));

	tbb::parallel_for(uint32_t(0), ntiles.x * ntiles.y, [&](uint32_t tile_id) {
		const glm::uvec2 tile = glm::uvec2(tile_id % ntiles.x, tile_id / ntiles.x);
		const glm::uvec2 tile_pos = tile * tile_size;
		const glm::uvec2 tile_end = glm::min(tile_pos + tile_size, fb_dims);
		const glm::uvec2 actual_tile_dims = tile_end - tile_pos;
		std::vector<float> tile_data(actual_tile_dims.x * actual_tile_dims.y * 3, 0.f);

		// TODO: Trace ray streams, generate streams and shade from ISPC for vectorization
		for (uint32_t j = 0; j < actual_tile_dims.y; ++j) {
			for (uint32_t i = 0; i < actual_tile_dims.x; ++i) {
				const glm::vec2 px = glm::vec2(i + tile_pos.x + 0.5f,
						j + tile_pos.y + 0.5f) / glm::vec2(fb_dims);

				const glm::vec3 dir = glm::normalize(px.x * dir_du
						+ px.y * dir_dv + dir_top_left);
				RTCRay ray;
				ray.org_x = pos.x;
				ray.org_y = pos.y;
				ray.org_z = pos.z;

				ray.dir_x = dir.x;
				ray.dir_y = dir.y;
				ray.dir_z = dir.z;

				ray.tnear = 0.f;
				ray.tfar = std::numeric_limits<float>::infinity();
				ray.time = 0.f;

				ray.mask = std::numeric_limits<uint32_t>::max();
				ray.id = 0;
				ray.flags = 0;

				RTCRayHit ray_hit;
				ray_hit.ray = ray;
				ray_hit.hit.primID = std::numeric_limits<uint32_t>::max();
				ray_hit.hit.geomID = std::numeric_limits<uint32_t>::max();
				ray_hit.hit.instID[0] = std::numeric_limits<uint32_t>::max();

				rtcIntersect1(scene, &context, &ray_hit);

				if (ray_hit.hit.geomID != std::numeric_limits<uint32_t>::max()) {
					const glm::uvec3 tri(indices[ray_hit.hit.primID]);

					const glm::vec3 v0(verts[tri.x]);
					const glm::vec3 v1(verts[tri.y]);
					const glm::vec3 v2(verts[tri.z]);

					glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
					n = (n + glm::vec3(1.f)) * 0.5f;

					const uint32_t pixel = (j * actual_tile_dims.x + i) * 3;
					tile_data[pixel] = n.x;
					tile_data[pixel + 1] = n.y;
					tile_data[pixel + 2] = n.z;
				}
			}
		}
		ispc::tile_to_uint8(tile_data.data(), color, fb_dims.x, fb_dims.y,
				tile_pos.x, tile_pos.y, actual_tile_dims.x, actual_tile_dims.y);
	});
}

