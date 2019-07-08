#include <vector>
#include <algorithm>
#include "util.h"
#include "scene.h"
#include "tiny_obj_loader.h"
#include "stb_image.h"

Scene Scene::load_obj(const std::string &file) {
	std::cout << "Loading OBJ: " << file << "\n";

	Scene scene;

	std::vector<uint32_t> material_ids;
	std::vector<DisneyMaterial> materials;
	size_t total_tris = 0;
	// Load the model w/ tinyobjloader. We just take any OBJ groups etc. stuff
	// that may be in the file and dump them all into a single OBJ model.
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> obj_materials;
	std::string err, warn;
#ifdef _WIN32
	const std::string mtl_base_dir = file.substr(0, file.rfind('\\'));
#else
	const std::string mtl_base_dir = file.substr(0, file.rfind('/'));
#endif
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &obj_materials, &warn, &err,
			file.c_str(), mtl_base_dir.c_str());
	if (!warn.empty()) {
		std::cout << "Warning loading '" << file << "': " << warn << "\n";
	}
	if (!ret || !err.empty()) {
		throw std::runtime_error("Error loading '" + file "': " + err);
	}

	for (size_t s = 0; s < shapes.size(); ++s) {
		// We load with triangulate on so we know the mesh will be all triangle faces
		const tinyobj::mesh_t &obj_mesh = shapes[s].mesh;

		// We've got to remap from 3 indices per-vert (independent for pos, normal & uv) used by
		// tinyobjloader over to single index per-vert (single for pos, normal & uv tuple) used by renderers
		std::map<glm::uvec3, uint32_t> index_mapping;
		Mesh mesh;
		// Note: not supporting per-primitive materials
		mesh.material_id = obj_mesh.material_ids[0];

		for (size_t f = 0; f < obj_mesh.num_face_vertices.size(); ++f) {
			if (obj_mesh.num_face_vertices[f] != 3) {
				throw std::runtime_error("Non-triangle face found in " + file + "-" + obj_mesh.name);
			}

			for (size_t i = 0; i < 3; ++i) {
				const glm::uvec3 idx(obj_mesh.indices[f * 3 + i].vertex_index,
						obj_mesh.indices[f * 3 + i].normal_index,
						obj_mesh.indices[f * 3 + i].texcoord_index);
				uint32_t vert_idx = 0;
				auto fnd = std::find(index_mapping.begin(), index_mapping.end(), idx);
				if (fnd != index_mapping.end()) {
					vert_idx = fnd->second;
				} else {
					vert_idx = mesh.vertices.size();
					index_mapping[idx] = vert_idx;

					mesh.vertices.emplace_back(attrib.vertices[3 * idx.vertex_index],
							attrib.vertices[3 * idx.vertex_index + 1],
							attrib.vertices[3 * idx.vertex_index + 2]);

					if (idx.normal_index != -1) {
						mesh.normals.emplace_back(attrib.normals[3 * idx.normal_index],
								attrib.normals[3 * idx.normal_index + 1],
								attrib.normals[3 * idx.normal_index + 2]);
					}

					if (idx.texcoord_index != -1) {
						mesh.uvs.emplace_back(attrib.texcoords[3 * idx.texcoord_index],
								attrib.texcoords[3 * idx.texcoord_index + 1]);
					}
				}
			}
		}
		total_tris += mesh.num_tris();
		scene.meshes.push_back(std::move(mesh));
	}
	std::cout << args[2] << " has " << total_tris << " tris, used over "
		<< shapes.size() << " shapes\n";

	// Parse the materials over to a similar DisneyMaterial representation
	for (const auto &m : obj_materials) {
		DisneyMaterial d;
		d.base_color = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
		d.specular = glm::clamp(m.shininess / 500.f, 0.f, 1.f);
		d.roughness = 1.f - d.specular;
		d.specular_transmission = glm::clamp(1.f - m.dissolve, 0.f, 1.f);

		if (!m.diffuse_texname.empty()) {
			scene.textures.emplace_back(m.diffuse_texname);
		}

		scene.materials.push_back(d);
	}

	const bool need_default_mat =
		std::find_if(meshes.begin(), meshes.end(),
				[](const Mesh &m) { return m.material_id == uint32_t(-1); }) != material_ids.end();
	if (need_default_mat) {
		std::cout << "No materials assigned for some or all objects, generating a default material\n";
		const uint32_t default_mat_id = scene.materials.size();
		scene.materials.push_back(DisneyMaterial());
		for (auto &m : meshes) {
			if (m.material_id == uint32_t(-1)) {
				m.material_id = default_mat_id;
			}
		}
	}

	return scene;
}

