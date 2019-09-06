#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "lights.h"
#include "material.h"
#include "mesh.h"

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<DisneyMaterial> materials;
    std::vector<Image> textures;
    std::vector<QuadLight> lights;

    Scene(const std::string &fname);

    size_t total_tris() const;

private:
    void load_obj(const std::string &file);

    void load_gltf(const std::string &file);
};

