#include <limits>
#include <iostream>
#include <tbb/parallel_for.h>
#include "render_embree.h"
#include "render_embree_ispc.h"

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

// TODO WILL: We might need to be careful of the alignment
// of the vector data
struct RaySoA {
	std::vector<float> org_x;
	std::vector<float> org_y;
	std::vector<float> org_z;
	std::vector<float> tnear;

	std::vector<float> dir_x;
	std::vector<float> dir_y;
	std::vector<float> dir_z;
	std::vector<float> time;

	std::vector<float> tfar;

	std::vector<unsigned int> mask;
	std::vector<unsigned int> id;
	std::vector<unsigned int> flags;

	RaySoA(const size_t nrays)
		: org_x(nrays, 0.f), org_y(nrays, 0.f), org_z(nrays, 0.f), tnear(nrays, 0.f),
		dir_x(nrays, 0.f), dir_y(nrays, 0.f), dir_z(nrays, 0.f), time(nrays, 0.f),
		tfar(nrays, std::numeric_limits<float>::infinity()),
		mask(nrays, std::numeric_limits<uint32_t>::max()),
		id(nrays, 0),
		flags(nrays, 0)
	{}
};

struct HitSoA {
	std::vector<float> ng_x;
	std::vector<float> ng_y;
	std::vector<float> ng_z;

	std::vector<float> u;
	std::vector<float> v;

	std::vector<unsigned int> prim_id;
	std::vector<unsigned int> geom_id;
	std::vector<unsigned int> inst_id;

	HitSoA(const size_t nrays)
		: ng_x(nrays, 0.f), ng_y(nrays, 0.f), ng_z(nrays, 0.f),
		u(nrays, 0.f), v(nrays, 0.f),
		prim_id(nrays, std::numeric_limits<uint32_t>::max()),
		geom_id(nrays, std::numeric_limits<uint32_t>::max()),
		inst_id(nrays, std::numeric_limits<uint32_t>::max())
	{}
};

RTCRayHitNp make_ray_hit_soa(RaySoA &rays, HitSoA &hits) {
	RTCRayHitNp rh;
	rh.ray.org_x = rays.org_x.data();
	rh.ray.org_y = rays.org_y.data();
	rh.ray.org_z = rays.org_z.data();
	rh.ray.tnear = rays.tnear.data();

	rh.ray.dir_x = rays.dir_x.data();
	rh.ray.dir_y = rays.dir_y.data();
	rh.ray.dir_z = rays.dir_z.data();
	rh.ray.time = rays.time.data();
	rh.ray.tfar = rays.tfar.data();

	rh.ray.mask = rays.mask.data();
	rh.ray.id = rays.id.data();
	rh.ray.flags = rays.flags.data();

	rh.hit.Ng_x = hits.ng_x.data();
	rh.hit.Ng_y = hits.ng_y.data();
	rh.hit.Ng_z = hits.ng_z.data();

	rh.hit.u = hits.u.data();
	rh.hit.v = hits.v.data();

	rh.hit.primID = hits.prim_id.data();
	rh.hit.geomID = hits.geom_id.data();
	rh.hit.instID[0] = hits.inst_id.data();
	return rh;
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

		const size_t npixels = actual_tile_dims.x * actual_tile_dims.y;
		std::vector<float> tile_data(npixels * 3, 0.f);

		RaySoA rays(npixels);
		HitSoA hits(npixels);
		RTCRayHitNp ray_hit = make_ray_hit_soa(rays, hits);

		// TODO: Make some shared ISPC state to store these params
		ispc::generate_primary_rays((ispc::RTCRayHitNp*)&ray_hit, tile_pos.x, tile_pos.y,
				fb_dims.x, fb_dims.y,
				actual_tile_dims.x, actual_tile_dims.y,
				&pos.x, &dir_du.x, &dir_dv.x, &dir_top_left.x);

		rtcIntersectNp(scene, &context, &ray_hit, npixels);

		ispc::shade_ray_stream((ispc::RTCRayHitNp*)&ray_hit, actual_tile_dims.x, actual_tile_dims.y,
				reinterpret_cast<const uint32_t*>(indices.data()),
				reinterpret_cast<const float*>(verts.data()),
				tile_data.data());

		ispc::tile_to_uint8(tile_data.data(), color, fb_dims.x, fb_dims.y,
				tile_pos.x, tile_pos.y, actual_tile_dims.x, actual_tile_dims.y);
	});
}

