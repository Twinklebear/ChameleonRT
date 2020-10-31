#include "flatten_gltf.h"
#include <iterator>
#include <string>
#include "tiny_gltf.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

glm::mat4 read_node_transform(const tinygltf::Node &n)
{
    glm::mat4 transform(1.f);
    if (!n.matrix.empty()) {
        transform = glm::make_mat4(n.matrix.data());
    } else {
        if (!n.scale.empty()) {
            transform = glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
        }
        if (!n.rotation.empty()) {
            const glm::quat rot =
                glm::quat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
            transform = glm::mat4_cast(rot) * transform;
        }
        if (!n.translation.empty()) {
            const glm::mat4 translate = glm::translate(
                glm::vec3(n.translation[0], n.translation[1], n.translation[2]));
            transform = translate * transform;
        }
    }
    return transform;
}

bool gltf_is_single_level(const tinygltf::Model &model)
{
    // Recursively traverse the GLTF scene graph and break once we
    // encounter multi-level instancing
    for (const auto &n : model.scenes[model.defaultScene].nodes) {
        const auto &node = model.nodes[n];
        if (!node.children.empty()) {
            return false;
        }
    }
    return true;
}

void flatten_gltf_node(const tinygltf::Node &node,
                       const tinygltf::Model &model,
                       const glm::mat4 &parent_transform,
                       std::vector<tinygltf::Node> &flattened)
{
    const glm::mat4 transform = parent_transform * read_node_transform(node);
    tinygltf::Node flat = node;
    flat.children.clear();
    // Clear the old transform and set the new flattened one
    flat.scale.clear();
    flat.rotation.clear();
    flat.translation.clear();
    flat.matrix.clear();
    std::copy(glm::value_ptr(transform),
              glm::value_ptr(transform) + 16,
              std::back_inserter(flat.matrix));
    if (flat.mesh != -1 || flat.camera != -1 || flat.skin != -1) {
        flattened.push_back(flat);
    }

    for (const auto &c : node.children) {
        flatten_gltf_node(model.nodes[c], model, transform, flattened);
    }
}

void flatten_gltf(tinygltf::Model &model)
{
    if (gltf_is_single_level(model)) {
        return;
    }

    // If the GLTF model is not single-level we'll make a new scene which
    // is single level, and set this as the new default scene
    std::vector<tinygltf::Node> flattened_nodes;
    for (const auto &n : model.scenes[model.defaultScene].nodes) {
        glm::mat4 transform(1.f);
        flatten_gltf_node(model.nodes[n], model, transform, flattened_nodes);
    }

    const size_t flat_scene_start = model.nodes.size();
    std::copy(flattened_nodes.begin(), flattened_nodes.end(), std::back_inserter(model.nodes));

    tinygltf::Scene flattened_scene = model.scenes[model.defaultScene];
    flattened_scene.name += "_flattened";
    flattened_scene.nodes.clear();
    for (size_t i = 0; i < flattened_nodes.size(); ++i) {
        flattened_scene.nodes.push_back(flat_scene_start + i);
    }
    model.defaultScene = model.scenes.size();
    model.scenes.push_back(flattened_scene);
}
