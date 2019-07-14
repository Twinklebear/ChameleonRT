#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "material.h"
#include "mesh.h"
#include "lights.h"

struct Scene {
	std::vector<Mesh> meshes;
	std::vector<DisneyMaterial> materials;
	std::vector<Image> textures;
	std::vector<QuadLight> lights;

	static Scene load_obj(const std::string &file);

	size_t total_tris() const;
};

