#include <limits>
#include <chrono>
#include <tbb/parallel_for.h>
#include "render_embree.h"
#include "render_embree_ispc.h"

RaySoA::RaySoA(const size_t nrays)
	: org_x(nrays, 0.f), org_y(nrays, 0.f), org_z(nrays, 0.f), tnear(nrays, 0.f),
	dir_x(nrays, 0.f), dir_y(nrays, 0.f), dir_z(nrays, 0.f), time(nrays, 0.f),
	tfar(nrays, std::numeric_limits<float>::infinity()),
	mask(nrays, std::numeric_limits<uint32_t>::max()),
	id(nrays, 0),
	flags(nrays, 0)
{}

void RaySoA::resize(const size_t nrays) {
	org_x.resize(nrays, 0.f);
	org_y.resize(nrays, 0.f);
	org_z.resize(nrays, 0.f);
	tnear.resize(nrays, 0.f);
	dir_x.resize(nrays, 0.f);
	dir_y.resize(nrays, 0.f);
	dir_z.resize(nrays, 0.f);
	time.resize(nrays, 0.f);
	tfar.resize(nrays, std::numeric_limits<float>::infinity());
	mask.resize(nrays, std::numeric_limits<uint32_t>::max());
	id.resize(nrays, 0);
	flags.resize(nrays, 0);

}

HitSoA::HitSoA(const size_t nrays)
	: ng_x(nrays, 0.f), ng_y(nrays, 0.f), ng_z(nrays, 0.f),
	u(nrays, 0.f), v(nrays, 0.f),
	prim_id(nrays, std::numeric_limits<uint32_t>::max()),
	geom_id(nrays, std::numeric_limits<uint32_t>::max()),
	inst_id(nrays, std::numeric_limits<uint32_t>::max())
{}

void HitSoA::resize(const size_t nrays) {
	ng_x.resize(nrays, 0.f);
	ng_y.resize(nrays, 0.f);
	ng_z.resize(nrays, 0.f);
	u.resize(nrays, 0.f);
	v.resize(nrays, 0.f);
	prim_id.resize(nrays, std::numeric_limits<uint32_t>::max());
	geom_id.resize(nrays, std::numeric_limits<uint32_t>::max());
	inst_id.resize(nrays, std::numeric_limits<uint32_t>::max());
}

RenderEmbree::RenderEmbree()
	: device(rtcNewDevice(NULL)), scene(rtcNewScene(device))
{}

void RenderEmbree::initialize(const int fb_width, const int fb_height) {
	fb_dims = glm::ivec2(fb_width, fb_height);
	img.resize(fb_width * fb_height);

	const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
			fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));
	tiles.resize(ntiles.x * ntiles.y);
	primary_rays.resize(tiles.size());
	for (size_t i = 0; i < tiles.size(); ++i) {
		tiles[i].resize(tile_size.x * tile_size.y * 3, 0.f);
		primary_rays[i].first.resize(tile_size.x * tile_size.y);
		primary_rays[i].second.resize(tile_size.x * tile_size.y);
	}
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

struct ViewParams {
	glm::vec3 pos, dir_du, dir_dv, dir_top_left;
	uint32_t frame_id;
};

struct Scene {
	RTCScene scene;
	RTCIntersectContext *coherent_context;
	RTCIntersectContext *incoherent_context;
};

struct Tile {
	uint32_t x, y;
	uint32_t width, height;
	uint32_t fb_width, fb_height;
	float *data;
};

double RenderEmbree::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;

	if (camera_changed) {
		frame_id = 0;
	}

	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(fb_dims.x) / fb_dims.y;

	ViewParams view_params;
	view_params.pos = pos;
	view_params.dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	view_params.dir_dv = glm::normalize(glm::cross(view_params.dir_du, dir)) * img_plane_size.y;
	view_params.dir_top_left = dir - 0.5f * view_params.dir_du - 0.5f * view_params.dir_dv;
	view_params.frame_id = frame_id;

	RTCIntersectContext coherent, incoherent;
	rtcInitIntersectContext(&coherent);
	coherent.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	rtcInitIntersectContext(&incoherent);
	incoherent.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

	Scene ispc_scene;
	ispc_scene.scene = scene;
	ispc_scene.coherent_context = &coherent;
	ispc_scene.incoherent_context = &incoherent;

	uint8_t *color = reinterpret_cast<uint8_t*>(img.data());
	std::fill(img.begin(), img.end(), 0);

	// Round up the number of tiles we need to run in case the
	// framebuffer is not an even multiple of tile size
	const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
			fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));

	auto start = high_resolution_clock::now();

	tbb::parallel_for(uint32_t(0), ntiles.x * ntiles.y, [&](uint32_t tile_id) {
		const glm::uvec2 tile = glm::uvec2(tile_id % ntiles.x, tile_id / ntiles.x);
		const glm::uvec2 tile_pos = tile * tile_size;
		const glm::uvec2 tile_end = glm::min(tile_pos + tile_size, fb_dims);
		const glm::uvec2 actual_tile_dims = tile_end - tile_pos;

		Tile ispc_tile;
		ispc_tile.x = tile_pos.x;
		ispc_tile.y = tile_pos.y;
		ispc_tile.width = actual_tile_dims.x;
		ispc_tile.height = actual_tile_dims.y;
		ispc_tile.fb_width = fb_dims.x;
		ispc_tile.fb_height = fb_dims.y;
		ispc_tile.data = tiles[tile_id].data();

		RTCRayHitNp ray_hit = make_ray_hit_soa(primary_rays[tile_id].first, primary_rays[tile_id].second);

		ispc::trace_rays(&ispc_scene, (ispc::RTCRayHitNp*)&ray_hit,
				&ispc_tile, &view_params,
				reinterpret_cast<const uint32_t*>(indices.data()),
				reinterpret_cast<const float*>(verts.data()));

		ispc::tile_to_uint8(&ispc_tile, color);
	});

	auto end = high_resolution_clock::now();
	const double render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-9;

	++frame_id;

	return fb_dims.x * fb_dims.y / render_time;
}

