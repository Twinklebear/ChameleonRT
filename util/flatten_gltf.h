#pragma once

#include <string>
#include "tiny_gltf.h"
#include <glm/glm.hpp>

glm::mat4 read_node_transform(const tinygltf::Node &n);

// Check if the GLTF scene contains only single-level instancing
bool gltf_is_single_level(const tinygltf::Model &model);

/* Convert the potentially multi-level instanced GLTf scene to one which uses only
 * single level instancing.
 */
void flatten_gltf(tinygltf::Model &model);
