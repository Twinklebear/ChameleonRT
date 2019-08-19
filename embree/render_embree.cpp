#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <tbb/parallel_for.h>
#include <glm/ext.hpp>
#include "render_embree.h"
#include "render_embree_ispc.h"

RenderEmbree::RenderEmbree() {
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	device = rtcNewDevice(NULL);
}

void RenderEmbree::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
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

void RenderEmbree::set_scene(const Scene &scene_data) { 
	frame_id = 0;

	std::vector<std::shared_ptr<embree::Instance>> instances;
	for (const auto &m : scene_data.meshes) {
		auto trimesh = std::make_shared<embree::TriangleMesh>(device, m.vertices, m.indices, m.normals, m.uvs);

		instances.push_back(std::make_shared<embree::Instance>(device, trimesh, m.material_id, glm::mat4(1.f)));
	}
	scene = std::make_shared<embree::TopLevelBVH>(device, instances);

	textures = scene_data.textures;
	ispc_textures.reserve(textures.size());
	std::transform(textures.begin(), textures.end(), std::back_inserter(ispc_textures),
			[](const Image &img) { return embree::ISPCTexture2D(img); });

	material_params.reserve(scene_data.materials.size());
	for (const auto &m : scene_data.materials) {
		embree::MaterialParams p;

		p.base_color = m.base_color;
		p.metallic = m.metallic;
		p.specular = m.specular;
		p.roughness = m.roughness;
		p.specular_tint = m.specular_tint;
		p.anisotropy = m.anisotropy;
		p.sheen = m.sheen;
		p.sheen_tint = m.sheen_tint;
		p.clearcoat = m.clearcoat;
		p.clearcoat_gloss = m.clearcoat_gloss;
		p.ior = m.ior;
		p.specular_transmission = m.specular_transmission;

		if (m.color_tex_id != -1) {
			p.color_texture = &ispc_textures[m.color_tex_id];
		} else {
			p.color_texture = nullptr;
		}

		material_params.push_back(p);
	}

	lights = scene_data.lights;
}

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

	embree::ViewParams view_params;
	view_params.pos = pos;
	view_params.dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	view_params.dir_dv = glm::normalize(glm::cross(view_params.dir_du, dir)) * img_plane_size.y;
	view_params.dir_top_left = dir - 0.5f * view_params.dir_du - 0.5f * view_params.dir_dv;
	view_params.frame_id = frame_id;

	embree::SceneContext ispc_scene;
	ispc_scene.scene = scene->handle;
	ispc_scene.instances = scene->ispc_instances.data();
	ispc_scene.materials = material_params.data();
	ispc_scene.lights = lights.data();
	ispc_scene.num_lights = lights.size();

	// Round up the number of tiles we need to run in case the
	// framebuffer is not an even multiple of tile size
	const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
			fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));

	auto start = high_resolution_clock::now();

	uint8_t *color = reinterpret_cast<uint8_t*>(img.data());
	tbb::parallel_for(uint32_t(0), ntiles.x * ntiles.y, [&](uint32_t tile_id) {
		const glm::uvec2 tile = glm::uvec2(tile_id % ntiles.x, tile_id / ntiles.x);
		const glm::uvec2 tile_pos = tile * tile_size;
		const glm::uvec2 tile_end = glm::min(tile_pos + tile_size, fb_dims);
		const glm::uvec2 actual_tile_dims = tile_end - tile_pos;

		embree::Tile ispc_tile;
		ispc_tile.x = tile_pos.x;
		ispc_tile.y = tile_pos.y;
		ispc_tile.width = actual_tile_dims.x;
		ispc_tile.height = actual_tile_dims.y;
		ispc_tile.fb_width = fb_dims.x;
		ispc_tile.fb_height = fb_dims.y;
		ispc_tile.data = tiles[tile_id].data();

		RTCRayHitNp ray_hit = embree::make_ray_hit_soa(primary_rays[tile_id].first, primary_rays[tile_id].second);

		ispc::trace_rays(&ispc_scene, (ispc::RTCRayHitNp*)&ray_hit,
				&ispc_tile, &view_params);

		ispc::tile_to_uint8(&ispc_tile, color);
	});

	auto end = high_resolution_clock::now();
	const double render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-9;

	++frame_id;

	return fb_dims.x * fb_dims.y / render_time;
}

