#include "scene.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "buffer_view.h"
#include "file_mapping.h"
#include "flatten_gltf.h"
#include "gltf_types.h"
#include "json.hpp"
#include "phmap_utils.h"
#include "stb_image.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include "util.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace std {
template <>
struct hash<glm::uvec3> {
    size_t operator()(glm::uvec3 const &v) const
    {
        return phmap::HashState().combine(0, v.x, v.y, v.z);
    }
};
}

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
    } else if (ext == "crts") {
        load_crts(fname);
#ifdef PBRT_PARSER_ENABLED
    } else if (ext == "pbrt" || ext == "pbf") {
        load_pbrt(fname);
#endif
    } else {
        std::cout << "Unsupported file type '" << ext << "'\n";
        throw std::runtime_error("Unsupported file type " + ext);
    }
}

size_t Scene::unique_tris() const
{
    return std::accumulate(
        meshes.begin(), meshes.end(), 0, [](const size_t &n, const Mesh &m) {
            return n + m.num_tris();
        });
}

size_t Scene::total_tris() const
{
    return std::accumulate(
        instances.begin(), instances.end(), 0, [&](const size_t &n, const Instance &i) {
            return n + meshes[i.mesh_id].num_tris();
        });
}

size_t Scene::num_geometries() const
{
    return std::accumulate(
        meshes.begin(), meshes.end(), 0, [](const size_t &n, const Mesh &m) {
            return n + m.geometries.size();
        });
}

void Scene::load_obj(const std::string &file)
{
    std::cout << "Loading OBJ: " << file << "\n";

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
    std::vector<uint32_t> material_ids;
    for (size_t s = 0; s < shapes.size(); ++s) {
        // We load with triangulate on so we know the mesh will be all triangle faces
        const tinyobj::mesh_t &obj_mesh = shapes[s].mesh;

        // We've got to remap from 3 indices per-vert (independent for pos, normal & uv) used
        // by tinyobjloader over to single index per-vert (single for pos, normal & uv tuple)
        // used by renderers
        phmap::parallel_flat_hash_map<glm::uvec3, uint32_t> index_mapping;
        Geometry geom;
        // Note: not supporting per-primitive materials
        material_ids.push_back(obj_mesh.material_ids[0]);

        auto minmax_matid =
            std::minmax_element(obj_mesh.material_ids.begin(), obj_mesh.material_ids.end());
        if (*minmax_matid.first != *minmax_matid.second) {
            std::cout
                << "Warning: per-face material IDs are not supported, materials may look "
                   "wrong."
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

                    if (idx.y != uint32_t(-1)) {
                        glm::vec3 n(attrib.normals[3 * idx.y],
                                    attrib.normals[3 * idx.y + 1],
                                    attrib.normals[3 * idx.y + 2]);
                        geom.normals.push_back(glm::normalize(n));
                    }

                    if (idx.z != uint32_t(-1)) {
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

    // OBJ has a single "instance"
    instances.emplace_back(glm::mat4(1.f), 0, material_ids);

    phmap::parallel_flat_hash_map<std::string, int32_t> texture_ids;
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
                textures.emplace_back(obj_base_dir + "/" + path, m.diffuse_texname, SRGB);
            }
            const int32_t id = texture_ids[m.diffuse_texname];
            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            d.base_color.r = *reinterpret_cast<float *>(&tex_mask);
        }
        materials.push_back(d);
    }

    validate_materials();

    // OBJ will not have any lights in it, so just generate one
    std::cout << "Generating light for OBJ scene\n";
    QuadLight light;
    light.emission = glm::vec4(20.f);
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
    if (get_file_extension(fname) == "gltf") {
        ret = context.LoadASCIIFromFile(&model, &err, &warn, fname.c_str());
    } else {
        ret = context.LoadBinaryFromFile(&model, &err, &warn, fname.c_str());
    }

    if (!warn.empty()) {
        std::cout << "TinyGLTF loading: " << fname << " warnings: " << warn << "\n";
    }

    if (!ret || !err.empty()) {
        throw std::runtime_error("TinyGLTF Error loading " + fname + " error: " + err);
    }

    if (model.defaultScene == -1) {
        model.defaultScene = 0;
    }

    flatten_gltf(model);

    std::vector<std::vector<uint32_t>> mesh_material_ids;
    // Load the meshes
    for (auto &m : model.meshes) {
        Mesh mesh;
        std::vector<uint32_t> material_ids;
        for (auto &p : m.primitives) {
            Geometry geom;
            material_ids.push_back(p.material);

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

            // Note: GLTF can have multiple texture coordinates used by different textures
            // (owch) I don't plan to support this
            auto fnd = p.attributes.find("TEXCOORD_0");
            if (fnd != p.attributes.end()) {
                Accessor<glm::vec2> uv_accessor(model.accessors[fnd->second], model);
                for (size_t i = 0; i < uv_accessor.size(); ++i) {
                    geom.uvs.push_back(uv_accessor[i]);
                }
            }

#if 0
            fnd = p.attributes.find("NORMAL");
            if (fnd != p.attributes.end()) {
                Accessor<glm::vec3> normal_accessor(model.accessors[fnd->second], model);
                for (size_t i = 0; i < normal_accessor.size(); ++i) {
                    geom.normals.push_back(normal_accessor[i]);
                }
            }
#endif

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
        mesh_material_ids.push_back(material_ids);
        meshes.push_back(mesh);
    }

    // Load images
    for (const auto &img : model.images) {
        if (img.component != 4) {
            std::cout << "WILL: Check non-4 component image support\n";
        }
        if (img.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            std::cout << "Non-uchar images are not supported\n";
            throw std::runtime_error("Unsupported image pixel type");
        }

        Image texture;
        texture.name = img.name;
        texture.width = img.width;
        texture.height = img.height;
        texture.channels = img.component;
        texture.img = img.image;
        // Assume linear unless we find it used as a color texture
        texture.color_space = LINEAR;
        textures.push_back(texture);
    }

    // Load materials
    for (const auto &m : model.materials) {
        DisneyMaterial mat;
        mat.base_color.x = m.pbrMetallicRoughness.baseColorFactor[0];
        mat.base_color.y = m.pbrMetallicRoughness.baseColorFactor[1];
        mat.base_color.z = m.pbrMetallicRoughness.baseColorFactor[2];

        mat.metallic = m.pbrMetallicRoughness.metallicFactor;

        mat.roughness = m.pbrMetallicRoughness.roughnessFactor;

        if (m.pbrMetallicRoughness.baseColorTexture.index != -1) {
            const int32_t id =
                model.textures[m.pbrMetallicRoughness.baseColorTexture.index].source;
            textures[id].color_space = SRGB;

            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
        }
        // glTF: metallic is blue channel, roughness is green channel
        if (m.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
            const int32_t id =
                model.textures[m.pbrMetallicRoughness.metallicRoughnessTexture.index].source;
            textures[id].color_space = LINEAR;

            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            SET_TEXTURE_CHANNEL(tex_mask, 2);
            mat.metallic = *reinterpret_cast<float *>(&tex_mask);

            tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            SET_TEXTURE_CHANNEL(tex_mask, 1);
            mat.roughness = *reinterpret_cast<float *>(&tex_mask);
        }
        materials.push_back(mat);
    }

    for (const auto &nid : model.scenes[model.defaultScene].nodes) {
        const tinygltf::Node &n = model.nodes[nid];
        if (n.mesh != -1) {
            const glm::mat4 transform = read_node_transform(n);
            instances.emplace_back(transform, n.mesh, mesh_material_ids[n.mesh]);
        }
    }

    validate_materials();

    // Does GLTF have lights in the file? If one is missing we should generate one,
    // otherwise we can load them
    std::cout << "Generating light for GLTF scene\n";
    QuadLight light;
    light.emission = glm::vec4(20.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.5, -0.8, -0.5)), 0);
    light.position = -10.f * light.normal;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 5.f;
    light.height = 5.f;
    lights.push_back(light);
}

void Scene::load_crts(const std::string &file)
{
    using json = nlohmann::json;
    std::cout << "Loading CRTS " << file << "\n";

    auto mapping = std::make_shared<FileMapping>(file);
    const uint64_t json_header_size = *reinterpret_cast<const uint64_t *>(mapping->data());
    const uint64_t total_header_size = json_header_size + sizeof(uint64_t);
    json header =
        json::parse(mapping->data() + sizeof(uint64_t), mapping->data() + total_header_size);

    const uint8_t *data_base = mapping->data() + total_header_size;
    // Blender only supports a single geometry per-mesh so this works kind of like a blend of
    // GLTF and OBJ
    for (size_t i = 0; i < header["meshes"].size(); ++i) {
        auto &m = header["meshes"][i];

        Geometry geom;
        {
            const uint64_t view_id = m["positions"].get<uint64_t>();
            auto &v = header["buffer_views"][view_id];
            const DTYPE dtype = parse_dtype(v["type"]);
            BufferView view(data_base + v["byte_offset"].get<uint64_t>(),
                            v["byte_length"].get<uint64_t>(),
                            dtype_stride(dtype));
            Accessor<glm::vec3> accessor(view);
            geom.vertices = std::vector<glm::vec3>(accessor.begin(), accessor.end());
        }
        {
            const uint64_t view_id = m["indices"].get<uint64_t>();
            auto &v = header["buffer_views"][view_id];
            const DTYPE dtype = parse_dtype(v["type"]);
            BufferView view(data_base + v["byte_offset"].get<uint64_t>(),
                            v["byte_length"].get<uint64_t>(),
                            dtype_stride(dtype));
            Accessor<glm::uvec3> accessor(view);
            geom.indices = std::vector<glm::uvec3>(accessor.begin(), accessor.end());
        }
        if (m.find("texcoords") != m.end()) {
            const uint64_t view_id = m["texcoords"].get<uint64_t>();
            auto &v = header["buffer_views"][view_id];
            const DTYPE dtype = parse_dtype(v["type"]);
            BufferView view(data_base + v["byte_offset"].get<uint64_t>(),
                            v["byte_length"].get<uint64_t>(),
                            dtype_stride(dtype));
            Accessor<glm::vec2> accessor(view);
            geom.uvs = std::vector<glm::vec2>(accessor.begin(), accessor.end());
        }
#if 0
        if (m.find("normals") != m.end()) {
            const uint64_t view_id = m["normals"].get<uint64_t>();
            auto &v = header["buffer_views"][view_id];
            const DTYPE dtype = parse_dtype(v["type"]);
            BufferView view(data_base + v["byte_offset"].get<uint64_t>(),
                            v["byte_length"].get<uint64_t>(),
                            dtype_stride(dtype));
            Accessor<glm::vec3> accessor(view);
            geom.normals = std::vector<glm::vec3>(accessor.begin(), accessor.end());
        }
#endif

        Mesh mesh;
        mesh.geometries.push_back(geom);
        meshes.push_back(mesh);
    }

    for (size_t i = 0; i < header["images"].size(); ++i) {
        auto &img = header["images"][i];

        const uint64_t view_id = img["view"].get<uint64_t>();
        auto &v = header["buffer_views"][view_id];
        const DTYPE dtype = parse_dtype(v["type"]);
        BufferView view(data_base + v["byte_offset"].get<uint64_t>(),
                        v["byte_length"].get<uint64_t>(),
                        dtype_stride(dtype));
        Accessor<uint8_t> accessor(view);

        stbi_set_flip_vertically_on_load(1);
        int x, y, n;
        uint8_t *img_data =
            stbi_load_from_memory(accessor.begin(), accessor.size(), &x, &y, &n, 4);
        stbi_set_flip_vertically_on_load(0);
        if (!img_data) {
            std::cout << "Failed to load " << img["name"].get<std::string>() << " from view\n";
            throw std::runtime_error("Failed to load " + img["name"].get<std::string>());
        }

        ColorSpace color_space = SRGB;
        if (img["color_space"].get<std::string>() == "LINEAR") {
            color_space = LINEAR;
        }

        textures.emplace_back(img_data, x, y, 4, img["name"].get<std::string>(), color_space);
        stbi_image_free(img_data);
    }

    for (size_t i = 0; i < header["materials"].size(); ++i) {
        auto &m = header["materials"][i];

        DisneyMaterial mat;

        const auto base_color_data = m["base_color"].get<std::vector<float>>();
        mat.base_color = glm::make_vec3(base_color_data.data());
        if (m.find("base_color_texture") != m.end()) {
            const int32_t id = m["base_color_texture"].get<int32_t>();
            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, id);
            mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
        }

        auto parse_float_param = [&](const std::string &param, float &val) {
            val = m[param].get<float>();
            const std::string texture_name = param + "_texture";
            if (m.find(texture_name) != m.end()) {
                const int32_t id = m[texture_name]["texture"].get<int32_t>();
                const uint32_t channel = m[texture_name]["channel"].get<uint32_t>();
                uint32_t tex_mask = TEXTURED_PARAM_MASK;
                SET_TEXTURE_ID(tex_mask, id);
                SET_TEXTURE_CHANNEL(tex_mask, channel);
                val = *reinterpret_cast<float *>(&tex_mask);
            }
        };

        parse_float_param("metallic", mat.metallic);
        parse_float_param("specular", mat.specular);
        parse_float_param("roughness", mat.roughness);
        parse_float_param("specular_tint", mat.specular_tint);
        parse_float_param("anisotropic", mat.anisotropy);
        parse_float_param("sheen", mat.sheen);
        parse_float_param("sheen_tint", mat.sheen_tint);
        parse_float_param("clearcoat", mat.clearcoat);
        // TODO: May need to invert this param coming from Blender to give clearcoat gloss?
        // or does the disney gloss term = roughness?
        parse_float_param("clearcoat_roughness", mat.clearcoat_gloss);
        parse_float_param("ior", mat.ior);
        parse_float_param("transmission", mat.specular_transmission);
        materials.push_back(mat);
    }

    for (size_t i = 0; i < header["objects"].size(); ++i) {
        auto &n = header["objects"][i];
        const std::string type = n["type"];
        const glm::mat4 matrix = glm::make_mat4(n["matrix"].get<std::vector<float>>().data());
        if (type == "MESH") {
            const auto mat_id = std::vector<uint32_t>{n["material"].get<uint32_t>()};
            instances.emplace_back(matrix, n["mesh"].get<uint64_t>(), mat_id);
        } else if (type == "LIGHT") {
            QuadLight light;
            const auto color = glm::make_vec3(n["color"].get<std::vector<float>>().data());
            light.emission = glm::vec4(color * n["energy"].get<float>(), 1.f);
            light.position = glm::column(matrix, 3);
            light.normal = -glm::normalize(glm::column(matrix, 2));
            light.v_x = glm::normalize(glm::column(matrix, 0));
            light.v_y = glm::normalize(glm::column(matrix, 1));
            light.width = n["size"][0].get<float>();
            light.height = n["size"][1].get<float>();
            lights.push_back(light);
        } else if (type == "CAMERA") {
            Camera camera;
            camera.position = glm::column(matrix, 3);
            const glm::vec3 dir(glm::normalize(-glm::column(matrix, 2)));
            camera.center = camera.position + dir * 10.f;
            camera.up = glm::normalize(glm::column(matrix, 1));
            // TODO: Not sure on why I need to scale fovy down to match Blender,
            // it doesn't quite line up either but this is pretty close.
            camera.fov_y = n["fov_y"].get<float>() / 1.18f;
            cameras.push_back(camera);
        } else {
            throw std::runtime_error("Unsupported object type: not a mesh or camera?");
        }
    }

    validate_materials();

    if (lights.empty()) {
        // TODO: Should add support for other light types? Then autogenerate a directional
        // light?
        std::cout << "No lights found in scene, generating one\n";
        QuadLight light;
        light.emission = glm::vec4(10.f);
        light.normal = glm::vec4(glm::normalize(glm::vec3(0.5, -0.8, -0.5)), 0);
        light.position = -10.f * light.normal;
        ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
        light.width = 5.f;
        light.height = 5.f;
        lights.push_back(light);
    }
}

#ifdef PBRT_PARSER_ENABLED

void Scene::load_pbrt(const std::string &file)
{
    std::shared_ptr<pbrt::Scene> scene = nullptr;
    try {
        if (get_file_extension(file) == "pbrt") {
            scene = pbrt::importPBRT(file);
        } else {
            scene = pbrt::Scene::loadFrom(file);
        }

        if (!scene) {
            throw std::runtime_error("Failed to load PBRT scene from " + file);
        }

        scene->makeSingleLevel();
    } catch (const std::runtime_error &e) {
        std::cout << "Error loading PBRT scene " << file << "\n";
        throw e;
    }

    const std::string pbrt_base_dir = file.substr(0, file.rfind('/'));

    // TODO: The world can also have some top-level things we may need to load. But is this
    // common? Or does Ingo's make single level flatten these down to a shape?
    for (const auto &obj : scene->world->shapes) {
        if (obj->material) {
            std::cout << "Top level Mat: " << obj->material->toString() << "\n";
        }

        // What would these top-level textures be used for?
        if (!obj->textures.empty()) {
            std::cout << "top level textures " << obj->textures.size() << "\n";
            for (const auto &t : obj->textures) {
                auto img_tex = std::dynamic_pointer_cast<pbrt::ImageTexture>(t.second);
                if (img_tex) {
                    std::cout << "Image texture: " << t.first << " from file "
                              << img_tex->fileName << "\n";
                } else {
                    std::cout << "Unsupported non-image texture used by texture '" << t.first
                              << "'\n";
                }
            }
        }

        if (obj->areaLight) {
            std::cout << "Encountered area light\n";
        }

        if (pbrt::TriangleMesh::SP mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(obj)) {
            std::cout << "Found root level triangle mesh w/ " << mesh->index.size()
                      << " triangles: " << mesh->toString() << "\n";
        } else if (pbrt::QuadMesh::SP mesh = std::dynamic_pointer_cast<pbrt::QuadMesh>(obj)) {
            std::cout << "Encountered root level quadmesh (unsupported type). Will TODO maybe "
                         "triangulate\n";
        } else {
            std::cout << "un-handled root level geometry type : " << obj->toString()
                      << std::endl;
        }
    }

    // For PBRTv3 Each Mesh corresponds to a PBRT Object, consisting of potentially
    // multiple Shapes. This maps to a Mesh with multiple geometries, which can then be
    // instanced
    phmap::parallel_flat_hash_map<pbrt::Material::SP, size_t> pbrt_materials;
    phmap::parallel_flat_hash_map<pbrt::Texture::SP, size_t> pbrt_textures;
    phmap::parallel_flat_hash_map<std::string, size_t> pbrt_objects;
    for (const auto &inst : scene->world->instances) {
        // Note: Materials are per-shape, so we should parse them and the IDs when loading
        // the shapes

        // Check if this object has already been loaded for the instance
        auto fnd = pbrt_objects.find(inst->object->name);
        size_t mesh_id = -1;
        std::vector<uint32_t> material_ids;
        if (fnd == pbrt_objects.end()) {
            std::cout << "Loading newly encountered instanced object " << inst->object->name
                      << "\n";

            // TODO: materials for pbrt
            std::vector<Geometry> geometries;
            for (const auto &g : inst->object->shapes) {
                if (pbrt::TriangleMesh::SP mesh =
                        std::dynamic_pointer_cast<pbrt::TriangleMesh>(g)) {
                    std::cout << "Object triangle mesh w/ " << mesh->index.size()
                              << " triangles: " << mesh->toString() << "\n";

                    uint32_t material_id = -1;
                    if (mesh->material) {
                        material_id = load_pbrt_materials(mesh->material,
                                                          mesh->textures,
                                                          pbrt_base_dir,
                                                          pbrt_materials,
                                                          pbrt_textures);
                    }
                    material_ids.push_back(material_id);

                    Geometry geom;
                    geom.vertices.reserve(mesh->vertex.size());
                    std::transform(
                        mesh->vertex.begin(),
                        mesh->vertex.end(),
                        std::back_inserter(geom.vertices),
                        [](const pbrt::vec3f &v) { return glm::vec3(v.x, v.y, v.z); });

                    geom.indices.reserve(mesh->index.size());
                    std::transform(
                        mesh->index.begin(),
                        mesh->index.end(),
                        std::back_inserter(geom.indices),
                        [](const pbrt::vec3i &v) { return glm::ivec3(v.x, v.y, v.z); });

                    geom.uvs.reserve(mesh->texcoord.size());
                    std::transform(mesh->texcoord.begin(),
                                   mesh->texcoord.end(),
                                   std::back_inserter(geom.uvs),
                                   [](const pbrt::vec2f &v) { return glm::vec2(v.x, v.y); });

                    geometries.push_back(geom);
                } else if (pbrt::QuadMesh::SP mesh =
                               std::dynamic_pointer_cast<pbrt::QuadMesh>(g)) {
                    std::cout << "Encountered instanced quadmesh (unsupported type). Will "
                                 "TODO maybe triangulate\n";
                } else {
                    std::cout << "un-handled instanced geometry type : " << g->toString()
                              << std::endl;
                }
            }
            if (inst->object->instances.size() > 0) {
                std::cout
                    << "Warning: Potentially multilevel instancing is in the scene after "
                       "flattening?\n";
            }
            // Mesh only contains unsupported objects, skip it
            if (geometries.empty()) {
                std::cout << "WARNING: Instance contains only unsupported geometries, "
                             "skipping\n";
                continue;
            }
            mesh_id = meshes.size();
            pbrt_objects[inst->object->name] = meshes.size();
            meshes.emplace_back(geometries);
        } else {
            // TODO: The instance needs to get the material ids for its shapes if we found it
            mesh_id = fnd->second;
            for (const auto &g : inst->object->shapes) {
                if (pbrt::TriangleMesh::SP mesh =
                        std::dynamic_pointer_cast<pbrt::TriangleMesh>(g)) {
                    uint32_t material_id = -1;
                    if (mesh->material) {
                        material_id = load_pbrt_materials(mesh->material,
                                                          mesh->textures,
                                                          pbrt_base_dir,
                                                          pbrt_materials,
                                                          pbrt_textures);
                    }
                    material_ids.push_back(material_id);
                }
            }
        }

        glm::mat4 transform(1.f);
        transform[0] = glm::vec4(inst->xfm.l.vx.x, inst->xfm.l.vx.y, inst->xfm.l.vx.z, 0.f);
        transform[1] = glm::vec4(inst->xfm.l.vy.x, inst->xfm.l.vy.y, inst->xfm.l.vy.z, 0.f);
        transform[2] = glm::vec4(inst->xfm.l.vz.x, inst->xfm.l.vz.y, inst->xfm.l.vz.z, 0.f);
        transform[3] = glm::vec4(inst->xfm.p.x, inst->xfm.p.y, inst->xfm.p.z, 1.f);

        instances.emplace_back(transform, mesh_id, material_ids);
    }

    validate_materials();

    std::cout << "Generating light for PBRT scene, TODO Will: Load them from the file\n";
    QuadLight light;
    light.emission = glm::vec4(20.f);
    light.normal = glm::vec4(glm::normalize(glm::vec3(0.5, -0.8, -0.5)), 0);
    light.position = -10.f * light.normal;
    ortho_basis(light.v_x, light.v_y, glm::vec3(light.normal));
    light.width = 5.f;
    light.height = 5.f;
    lights.push_back(light);
}

uint32_t Scene::load_pbrt_materials(
    const pbrt::Material::SP &mat,
    const std::map<std::string, pbrt::Texture::SP> &texture_overrides,
    const std::string &pbrt_base_dir,
    phmap::parallel_flat_hash_map<pbrt::Material::SP, size_t> &pbrt_materials,
    phmap::parallel_flat_hash_map<pbrt::Texture::SP, size_t> &pbrt_textures)
{
    auto fnd = pbrt_materials.find(mat);
    if (fnd != pbrt_materials.end()) {
        return fnd->second;
    }

    // TODO: The way some of the texturing can work is that the object's attached textures
    // can override the material parameters in some way
    if (!texture_overrides.empty()) {
        std::cout << "TODO: per-shape texture override support\n";
        for (const auto &t : texture_overrides) {
            std::cout << t.first << "\n";
        }
    }

    DisneyMaterial loaded_mat;
    if (auto m = std::dynamic_pointer_cast<pbrt::DisneyMaterial>(mat)) {
        loaded_mat.anisotropy = m->anisotropic;
        loaded_mat.clearcoat = m->clearCoat;
        loaded_mat.clearcoat_gloss = m->clearCoatGloss;
        loaded_mat.base_color = glm::vec3(m->color.x, m->color.y, m->color.z);
        loaded_mat.ior = m->eta;
        loaded_mat.metallic = m->metallic;
        loaded_mat.roughness = m->roughness;
        loaded_mat.sheen = m->sheen;
        loaded_mat.sheen_tint = m->sheenTint;
        loaded_mat.specular_tint = m->specularTint;
        // PBRT Doesn't use the specular parameter or have textures for the Disney
        // Material?
        loaded_mat.specular = 0.f;
    } else if (auto m = std::dynamic_pointer_cast<pbrt::PlasticMaterial>(mat)) {
        loaded_mat.base_color = glm::vec3(m->kd.x, m->kd.y, m->kd.z);
        if (m->map_kd) {
            if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
                loaded_mat.base_color =
                    glm::vec3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
            } else {
                const uint32_t tex_id =
                    load_pbrt_texture(m->map_kd, pbrt_base_dir, pbrt_textures);
                if (tex_id != uint32_t(-1)) {
                    uint32_t tex_mask = TEXTURED_PARAM_MASK;
                    SET_TEXTURE_ID(tex_mask, tex_id);
                    loaded_mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
                }
            }
        }

        const glm::vec3 ks(m->ks.x, m->ks.y, m->ks.z);
        loaded_mat.specular = luminance(ks);
        loaded_mat.roughness = m->roughness;
    } else if (auto m = std::dynamic_pointer_cast<pbrt::MatteMaterial>(mat)) {
        loaded_mat.base_color = glm::vec3(m->kd.x, m->kd.y, m->kd.z);
        if (m->map_kd) {
            if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
                loaded_mat.base_color =
                    glm::vec3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
            } else {
                const uint32_t tex_id =
                    load_pbrt_texture(m->map_kd, pbrt_base_dir, pbrt_textures);
                if (tex_id != uint32_t(-1)) {
                    uint32_t tex_mask = TEXTURED_PARAM_MASK;
                    SET_TEXTURE_ID(tex_mask, tex_id);
                    loaded_mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
                }
            }
        }
    } else if (auto m = std::dynamic_pointer_cast<pbrt::SubstrateMaterial>(mat)) {
        loaded_mat.base_color = glm::vec3(m->kd.x, m->kd.y, m->kd.z);
        if (m->map_kd) {
            if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
                loaded_mat.base_color =
                    glm::vec3(const_tex->value.x, const_tex->value.y, const_tex->value.z);
            } else {
                const uint32_t tex_id =
                    load_pbrt_texture(m->map_kd, pbrt_base_dir, pbrt_textures);
                if (tex_id != uint32_t(-1)) {
                    uint32_t tex_mask = TEXTURED_PARAM_MASK;
                    SET_TEXTURE_ID(tex_mask, tex_id);
                    loaded_mat.base_color.r = *reinterpret_cast<float *>(&tex_mask);
                }
            }
        }
        // Sounds like this is kind of what the SubstrateMaterial acts like? Diffuse with a
        // specular and clearcoat?
        const glm::vec3 ks(m->ks.x, m->ks.y, m->ks.z);
        loaded_mat.specular = luminance(ks);
        loaded_mat.roughness = 1.f;
        loaded_mat.clearcoat = 1.f;
        loaded_mat.clearcoat_gloss = luminance(ks);
    } else {
        std::cout << "Unsupported material type " << mat->toString() << "\n";
        return -1;
    }

    const uint32_t mat_id = materials.size();
    pbrt_materials[mat] = mat_id;
    materials.push_back(loaded_mat);
    return mat_id;
}

uint32_t Scene::load_pbrt_texture(
    const pbrt::Texture::SP &texture,
    const std::string &pbrt_base_dir,
    phmap::parallel_flat_hash_map<pbrt::Texture::SP, size_t> &pbrt_textures)
{
    auto fnd = pbrt_textures.find(texture);
    if (fnd != pbrt_textures.end()) {
        return fnd->second;
    }

    if (auto t = std::dynamic_pointer_cast<pbrt::ImageTexture>(texture)) {
        std::string path = t->fileName;
        canonicalize_path(path);
        try {
            Image img(pbrt_base_dir + "/" + path, t->fileName, SRGB);
            const uint32_t id = textures.size();
            pbrt_textures[texture] = id;
            textures.push_back(img);
            std::cout << "Loaded image texture: " << t->fileName << "\n";
            return id;
        } catch (const std::runtime_error &) {
            std::cout << "Unsupported file format or failed to load file: " << t->fileName
                      << "\n";
            return -1;
        }
    }

    std::cout << "Texture type " << texture->toString() << " is not supported\n";
    return -1;
}

#endif

void Scene::validate_materials()
{
    const bool need_default_mat =
        std::find_if(instances.begin(), instances.end(), [](const Instance &i) {
            return std::find(i.material_ids.begin(), i.material_ids.end(), uint32_t(-1)) !=
                   i.material_ids.end();
        }) != instances.end();

    if (need_default_mat) {
        std::cout << "No materials assigned for some objects, generating a default\n";
        const uint32_t default_mat_id = materials.size();
        materials.push_back(DisneyMaterial());
        for (auto &i : instances) {
            for (auto &m : i.material_ids) {
                if (m == -1) {
                    m = default_mat_id;
                }
            }
        }
    }
}
