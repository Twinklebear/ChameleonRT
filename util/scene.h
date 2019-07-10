#pragma once

#include <string>
#include <unordered_map>
#include "material.h"
#include "mesh.h"

struct Scene {
	std::vector<Mesh> meshes;
	std::vector<DisneyMaterial> materials;
	std::unordered_map<std::string, Image> textures;

	static Scene load_obj(const std::string &file);
};

