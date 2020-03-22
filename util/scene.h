#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "camera.h"
#include "lights.h"
#include "material.h"
#include "mesh.h"
#include "phmap.h"

#ifdef PBRT_PARSER_ENABLED
#include "pbrtParser/Scene.h"
#endif

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    std::vector<DisneyMaterial> materials;
    std::vector<Image> textures;
    std::vector<QuadLight> lights;
    std::vector<Camera> cameras;

    Scene(const std::string &fname);
    Scene() = default;

    // Compute the unique number of triangles in the scene
    size_t unique_tris() const;

    // Compute the total number of triangles in the scene (after instancing)
    size_t total_tris() const;

    size_t num_geometries() const;

private:
    void load_obj(const std::string &file);

    void load_gltf(const std::string &file);

    void load_crts(const std::string &file);

#ifdef PBRT_PARSER_ENABLED
    void load_pbrt(const std::string &file);

    uint32_t load_pbrt_materials(
        const pbrt::Material::SP &mat,
        const std::map<std::string, pbrt::Texture::SP> &texture_overrides,
        const std::string &pbrt_base_dir,
        phmap::parallel_flat_hash_map<pbrt::Material::SP, size_t> &pbrt_materials,
        phmap::parallel_flat_hash_map<pbrt::Texture::SP, size_t> &pbrt_textures);

    uint32_t load_pbrt_texture(
        const pbrt::Texture::SP &texture,
        const std::string &pbrt_base_dir,
        phmap::parallel_flat_hash_map<pbrt::Texture::SP, size_t> &pbrt_textures);
#endif

    void validate_materials();
};
