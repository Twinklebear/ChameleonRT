#pragma once

#include <utility>
#include <embree3/rtcore.h>
#include "render_backend.h"

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

	RaySoA() = default;
	RaySoA(const size_t nrays);
	void resize(const size_t nrays);
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

	HitSoA() = default;
	HitSoA(const size_t nrays);
	void resize(const size_t nrays);
};


struct RenderEmbree : RenderBackend {
	RTCDevice device;
	RTCScene scene;
	glm::uvec2 fb_dims;
	std::vector<glm::vec4> verts;
	std::vector<glm::uvec3> indices;

	uint32_t frame_id = 0;
	glm::uvec2 tile_size = glm::uvec2(64);
	std::vector<std::vector<float>> tiles;
	std::vector<std::pair<RaySoA, HitSoA>> primary_rays;

	RenderEmbree();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;
};

