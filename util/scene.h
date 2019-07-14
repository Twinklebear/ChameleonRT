#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "material.h"
#include "mesh.h"

struct Scene {
	std::vector<Mesh> meshes;
	std::vector<DisneyMaterial> materials;
	std::vector<Image> textures;

	static Scene load_obj(const std::string &file);
};

