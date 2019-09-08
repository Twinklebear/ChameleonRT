#include "scene.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "buffer_view.h"
#include "gltf_types.h"
#include "stb_image.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include "util.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>

struct VertIdxLess {
    bool operator()(const glm::uvec3 &a, const glm::uvec3 &b) const
    {
        return a.x < b.x || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z);
    }
};

bool operator==(const glm::uvec3 &a, const glm::uvec3 &b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

Scene::Scene(const std::string &fname)
{
    const std::string ext = get_file_extension(fname);
    if (ext == "obj") {
        load_obj(fname);
    } else if (ext == "gltf" || ext == "glb") {
        load_gltf(fname);
    } else {
        std::cout << "Unsupported file type '" << ext << "'\n";
        throw std::runtime_error("Unsupported file type " + ext);
    }
}

size_t Scene::total_tris() const
{
    return std::accumulate(
        meshes.begin(), meshes.end(), size_t(0), [](const size_t &n, const Mesh &m) {
            return n + std::accumulate(
                           m.geometries.begin(),
                           m.geometries.end(),
                           size_t(0),
                           [](const size_t &n, const Geometry &g) { return n + g.num_tris(); });
        });
}

void Scene::load_obj(const std::string &file)
{
    std::cout << "Loading OBJ: " << file << "\n";

    std::vector<uint32_t> material_ids;
    // Load the model w/ tinyobjloader. We just take any OBJ groups etc. stuff
    // that may be in the file and dump them all into a single OBJ model.
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> obj_materials;
    std::string err, warn;
    const std::string obj_base_dir = file.substr(0, file.rfind('/'));
    bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &obj_materials, &warn, &err, file.c_str(), obj_base_dir.c_str());
    if (!warn.empty()) {
        std::cout << "TinyOBJ loading '" << file << "': " << warn << "\n";
    }
    if (!ret || !err.empty()) {
        throw std::runtime_error("TinyOBJ Error loading " + file + " error: " + err);
    }

    Mesh mesh;
    for (size_t s = 0; s < shapes.size(); ++s) {
        // We load with triangulate on so we know the mesh will be all triangle faces
        const tinyobj::mesh_t &obj_mesh = shapes[s].mesh;

        // We've got to remap from 3 indices per-vert (independent for pos, normal & uv) used by
        // tinyobjloader over to single index per-vert (single for pos, normal & uv tuple) used by
        // renderers
        std::map<glm::uvec3, uint32_t, VertIdxLess> index_mapping;
        Geometry geom;
        // Note: not supporting per-primitive materials
        geom.material_id = obj_mesh.material_ids[0];

        auto minmax_matid =
            std::minmax_element(obj_mesh.material_ids.begin(), obj_mesh.material_ids.end());
        if (*minmax_matid.first != *minmax_matid.second) {
            std::cout
                << "Warning: per-face material IDs are not supported, materials may look wrong."
                   " Please reexport your mesh with each material group as an OBJ group\n";
        }

        for (size_t f = 0; f < obj_mesh.num_face_vertices.size(); ++f) {
            if (obj_mesh.num_face_vertices[f] != 3) {
                throw std::runtime_error("Non-triangle face found in " + file + "-" +
                                         shapes[s].name);
            }

            glm::uvec3 tri_indices;
            for (size_t i = 0; i < 3; ++i) {
                const glm::uvec3 idx(obj_mesh.indices[f * 3 + i].vertex_index,
                                     obj_mesh.indices[f * 3 + i].normal_index,
                                     obj_mesh.indices[f * 3 + i].texcoord_index);
                uint32_t vert_idx = 0;
                auto fnd = index_mapping.find(idx);
                if (fnd != index_mapping.end()) {
                    vert_idx = fnd->second;
                } else {
                    vert_idx = geom.vertices.size();
                    index_mapping[idx] = vert_idx;

                    geom.vertices.emplace_back(attrib.vertices[3 * idx.x],
                                               attrib.vertices[3 * idx.x + 1],
                                               attrib.vertices[3 * idx.x + 2]);

                    if (idx.y != -1) {
                        glm::vec3 n(attrib.normals[3 * idx.y],
                                    attrib.normals[3 * idx.y + 1],
                                    attrib.normals[3 * idx.y + 2]);
                        geom.normals.push_back(glm::normalize(n));
                    }

                    if (idx.z != -1) {
                        geom.uvs.emplace_back(attrib.texcoords[2 * idx.z],
                                              attrib.texcoords[2 * idx.z + 1]);
                    }
                }
                tri_indices[i] = vert_idx;
            }
            geom.indices.push_back(tri_indices);
        }
        mesh.geometries.push_back(geom);
    }
    meshes.push_back(mesh);

    std::unordered_map<std::string, int32_t> texture_ids;
    // Parse the materials over to a similar DisneyMaterial representation
    for (const auto &m : obj_materials) {
        DisneyMaterial d;
        d.base_color = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
        d.specular = glm::clamp(m.shininess / 500.f, 0.f, 1.f);
        d.roughness = 1.f - d.specular;
        d.specular_transmission = glm::clamp(1.f - m.dissolve, 0.f, 1.f);

        if (!m.diffuse_texname.empty()) {
            std::string path = m.diffuse_texname;
            canonicalize_path(path);
            if (texture_ids.find(m.diffuse_texname) == texture_ids.end()) {
                texture_ids[m.diffuse_texname] = textures.size();
                textures.emplace_back(obj_base_dir + "/" + path, m.diffuse_texname);
            }
            d.color_tex_id = texture_ids[m.diffuse_texname];
        }
        materials.push_back(d);
    }

    const bool need_default_mat =
        std::find_if(meshes.begin(), meshes.end(), [](const Mesh &m) {
            return std::find_if(m.geometries.begin(), m.geometries.end(), [](const Geometry &g) {
                       return g.material_id == uint32_t(-1);
                   }) != m.geometries.end();
        }) != meshes.end();

    if (need_default_mat) {
        std::cout
            << "No materials assigned for some or all objects, generating a default material\n";
        const uint32_t default_mat_id = materials.size();
        materials.push_back(DisneyMaterial());

        // OBJ will have just one mesh, containg all geometries
        for (auto &g : meshes[0].geometries) {
            if (g.material_id == uint32_t(-1)) {
                g.material_id = default_mat_id;
            }
        }
    }

    // OBJ will not have any lights in it, so just generate one
    std::cout << "Generating light for OBJ scene\n";
    QuadLight light;
    light.emission = glm::vec4(5.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.5, -0.8, -0.5)), 0);
    light.position = -10.f * light.normal;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 5.f;
    light.height = 5.f;
    lights.push_back(light);
}

void Scene::load_gltf(const std::string &fname)
{
    std::cout << "Loading GLTF " << fname << "\n";

    tinygltf::Model model;
    tinygltf::TinyGLTF context;
    std::string err, warn;
    bool ret = false;
    if (get_file_extension(fname) == "glb") {
        ret = context.LoadBinaryFromFile(&model, &err, &warn, fname.c_str());
    } else {
        ret = context.LoadASCIIFromFile(&model, &err, &warn, fname.c_str());
    }

    if (!warn.empty()) {
        std::cout << "TinyGLTF loading: " << fname << " warnings: " << warn << "\n";
    }

    if (!ret || !err.empty()) {
        throw std::runtime_error("TinyGLTF Error loading " + fname + " error: " + err);
    }

    std::cout << "Default scene: " << model.defaultScene << "\n";

    std::cout << "# of scenes " << model.scenes.size() << "\n";
    for (const auto &scene : model.scenes) {
        std::cout << "Scene: " << scene.name << "\n";
    }

    // Load each prim of each mesh as its own mesh for testing
    for (auto &m : model.meshes) {
        Mesh mesh;
        for (auto &p : m.primitives) {
            Geometry geom;

            if (p.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cout << "Unsupported primitive mode! File must contain only triangles\n";
                throw std::runtime_error(
                    "Unsupported primitive mode! Only triangles are supported");
            }

            // Note: assumes there is a POSITION (is this required by the gltf spec?)
            Accessor<glm::vec3> pos_accessor(model.accessors[p.attributes["POSITION"]], model);
            for (size_t i = 0; i < pos_accessor.size(); ++i) {
                geom.vertices.push_back(pos_accessor[i]);
            }

            if (model.accessors[p.indices].componentType ==
                TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                Accessor<uint16_t> index_accessor(model.accessors[p.indices], model);
                for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
                    geom.indices.push_back(glm::uvec3(index_accessor[i * 3],
                                                      index_accessor[i * 3 + 1],
                                                      index_accessor[i * 3 + 2]));
                }
            } else if (model.accessors[p.indices].componentType ==
                       TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                Accessor<uint32_t> index_accessor(model.accessors[p.indices], model);
                for (size_t i = 0; i < index_accessor.size() / 3; ++i) {
                    geom.indices.push_back(glm::uvec3(index_accessor[i * 3],
                                                      index_accessor[i * 3 + 1],
                                                      index_accessor[i * 3 + 2]));
                }
            } else {
                std::cout << "Unsupported index type\n";
                throw std::runtime_error("Unsupported index component type");
            }

            mesh.geometries.push_back(geom);
        }
        meshes.push_back(mesh);
    }

    // TODO: Load materials if defined in the file
    const uint32_t default_mat_id = materials.size();
    materials.push_back(DisneyMaterial());
    for (auto &m : meshes) {
        for (auto &g : m.geometries) {
            if (g.material_id == uint32_t(-1)) {
                g.material_id = default_mat_id;
            }
        }
    }

    // Does GLTF have lights in the file? If one is missing we should generate one,
    // otherwise we can load them
    std::cout << "Generating light for GLTF scene\n";
    QuadLight light;
    light.emission = glm::vec4(5.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.5, -0.8, -0.5)), 0);
    light.position = -10.f * light.normal;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 5.f;
    light.height = 5.f;
    lights.push_back(light);
}
