#pragma once

#include <utility>
#include <memory>
#include <vector>
#include <embree3/rtcore.h>
#include "embree_utils.h"
#include "material.h"
#include "render_backend.h"

struct RenderEmbree : RenderBackend {
	RTCDevice device;
	glm::uvec2 fb_dims;

	std::vector<std::shared_ptr<embree::TriangleMesh>> meshes;
	std::shared_ptr<embree::TopLevelBVH> scene;
	std::vector<DisneyMaterial> materials;

	uint32_t frame_id = 0;
	glm::uvec2 tile_size = glm::uvec2(64);
	std::vector<std::vector<float>> tiles;
	std::vector<std::pair<embree::RaySoA, embree::HitSoA>> primary_rays;

	RenderEmbree();

	void initialize(const int fb_width, const int fb_height) override;
	void set_scene(const Scene &scene) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;
};

