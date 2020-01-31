#include "objects.h"
#include <unordered_map>
#include "texture_channel_mask.h"
#include <glm/gtc/type_ptr.hpp>

using namespace ospcommon::math;

namespace device {

void Light::commit()
{
    auto emission = getParam<vec3f>("color", vec3f(1.f)) * getParam<float>("intensity", 1.f);
    light.emission = glm::vec4(emission.x, emission.y, emission.z, 1.f);

    auto position = getParam<vec3f>("position", vec3f(0.f));
    light.position = glm::vec4(position.x, position.y, position.z, 1.f);

    auto edge1 = getParam<vec3f>("edge1", vec3f(1.f, 0.f, 0.f));
    auto edge2 = getParam<vec3f>("edge2", vec3f(0.f, 1.f, 0.f));
    light.width = length(edge1);
    light.height = length(edge2);

    edge1 = normalize(edge1);
    edge2 = normalize(edge2);
    light.v_x = glm::vec3(edge1.x, edge1.y, edge1.z);
    light.v_x = glm::vec3(edge2.x, edge2.y, edge2.z);

    auto normal = cross(edge1, edge2);
    light.normal = glm::vec4(normal.x, normal.y, normal.z, 1.f);
}

void Camera::commit()
{
    auto position = getParam<vec3f>("position", vec3f(0.f));
    camera.position = glm::vec3(position.x, position.y, position.z);

    auto direction = getParam<vec3f>("direction", vec3f(0.f, 0.f, -1.f));
    camera.center = camera.position + glm::vec3(direction.x, direction.y, direction.z);

    auto up = getParam<vec3f>("up", vec3f(0.f, 1.f, 0.f));
    camera.up = glm::vec3(up.x, up.y, up.z);

    camera.fov_y = getParam<float>("fovy", 55.f);
}

void Geometry::commit()
{
    vertices = getParam<Data *>("vertex.position", nullptr);
    if (!vertices) {
        throw std::runtime_error("vertex.position is required");
    }
    indices = getParam<Data *>("index", nullptr);
    if (!indices) {
        throw std::runtime_error("index is required");
    }

    normals = getParam<Data *>("vertex.normal", nullptr);
    uvs = getParam<Data *>("vertex.texcoord", nullptr);
}

GeometricModel::GeometricModel(Geometry *geom) : geometry(geom) {}

void GeometricModel::commit()
{
    material_id = getParam<uint32_t>("material", -1);
}

void Group::commit()
{
    geometry = getParam<Data *>("geometry", nullptr);
    if (!geometry) {
        throw std::runtime_error("geometry is required");
    }
}

void Texture::commit()
{
    format = getParam<int>("format", OSP_TEXTURE_RGB8);
    channels = format == OSP_TEXTURE_RGB8 ? 3 : 4;
    img = getParam<Data *>("data", nullptr);
    if (!img) {
        throw std::runtime_error("data is required!");
    }
}

void Material::commit()
{
    auto base_col = getParam<vec3f>("baseColor", vec3f(0.9f));
    base_color = glm::vec3(base_col.x, base_col.y, base_col.z);
    tex_base_color = getParam<Texture *>("map_baseColor", nullptr);

    metallic = getParam<float>("metallic", 0.f);
    specular = getParam<float>("specular", 0.f);
    roughness = getParam<float>("roughness", 1.f);
    specular_tint = getParam<float>("specular_tint", 0.f);
    anisotropy = getParam<float>("anisotropy", 0.f);
    sheen = getParam<float>("sheen", 0.f);
    sheen_tint = getParam<float>("sheen_tint", 0.f);
    clearcoat = getParam<float>("clearcoat", 0.f);
    clearcoat_gloss = getParam<float>("clearcoat_gloss", 0.f);
    ior = getParam<float>("ior", 0.f);
    specular_transmission = getParam<float>("specular_transmission", 0.f);
}

Instance::Instance(Group *group) : group(group) {}

void Instance::commit()
{
    affine3f xfm = getParam<affine3f>("xfm", affine3f{});
    transform = glm::mat4(glm::make_mat4x3(reinterpret_cast<float *>(&xfm)));
}

void World::commit()
{
    Data *instance_data = getParam<Data *>("instance", nullptr);
    if (!instance_data) {
        throw std::runtime_error("instance is required");
    }
    Data *lights = getParam<Data *>("light", nullptr);
    if (!lights) {
        throw std::runtime_error("light is required");
    }

    scene.lights.clear();
    std::transform(reinterpret_cast<Light *>(lights->begin()),
                   reinterpret_cast<Light *>(lights->end()),
                   std::back_inserter(scene.lights),
                   [](const Light &l) { return l.light; });

    scene.meshes.clear();
    scene.instances.clear();
    Instance *instances = reinterpret_cast<Instance *>(instance_data->data());
    std::unordered_map<Group *, uint32_t> mesh_ids;
    for (size_t i = 0; i < instance_data->size().x; ++i) {
        ::Instance inst;
        inst.transform = instances[i].transform;
        Group *g = instances[i].group;
        auto fnd = mesh_ids.find(g);
        if (fnd == mesh_ids.end()) {
            inst.mesh_id = scene.meshes.size();
            mesh_ids[g] = inst.mesh_id;

            Mesh mesh;
            GeometricModel *geom_models =
                reinterpret_cast<GeometricModel *>(g->geometry->data());
            for (size_t j = 0; j < g->geometry->size().x; ++j) {
                Geometry *g = geom_models[j].geometry;
                ::Geometry mesh_g;
                mesh_g.vertices =
                    std::vector<glm::vec3>(reinterpret_cast<glm::vec3 *>(g->vertices->begin()),
                                           reinterpret_cast<glm::vec3 *>(g->vertices->end()));

                mesh_g.indices = std::vector<glm::uvec3>(
                    reinterpret_cast<glm::uvec3 *>(g->indices->begin()),
                    reinterpret_cast<glm::uvec3 *>(g->indices->end()));

                if (g->uvs) {
                    mesh_g.uvs =
                        std::vector<glm::vec2>(reinterpret_cast<glm::vec2 *>(g->uvs->begin()),
                                               reinterpret_cast<glm::vec2 *>(g->uvs->end()));
                }
                if (g->normals) {
                    mesh_g.normals = std::vector<glm::vec3>(
                        reinterpret_cast<glm::vec3 *>(g->normals->begin()),
                        reinterpret_cast<glm::vec3 *>(g->normals->end()));
                }
            }
            scene.meshes.push_back(mesh);
        } else {
            inst.mesh_id = fnd->second;
        }

        GeometricModel *geom_models = reinterpret_cast<GeometricModel *>(g->geometry->data());
        for (size_t j = 0; j < g->geometry->size().x; ++j) {
            inst.material_ids.push_back(geom_models[j].material_id);
        }
        scene.instances.push_back(inst);
    }
}

void Renderer::commit()
{
    Data *mat_data = getParam<Data *>("material", nullptr);
    if (!mat_data) {
        throw std::runtime_error("materials are required!");
    }

    images.clear();
    materials.clear();

    Material *mats = reinterpret_cast<Material *>(mat_data->data());
    std::unordered_map<Texture *, uint32_t> texture_ids;
    for (size_t i = 0; i < mat_data->size().x; ++i) {
        DisneyMaterial m;
        m.base_color = mats[i].base_color;
        if (mats[i].tex_base_color) {
            Texture *tex = mats[i].tex_base_color;
            auto fnd = texture_ids.find(tex);
            uint32_t tex_id = -1;
            if (fnd == texture_ids.end()) {
                tex_id = images.size();
                texture_ids[tex] = tex_id;

                Image img;
                img.name = "device generated";
                img.width = tex->img->size().x;
                img.height = tex->img->size().y;
                img.channels = tex->channels;
                img.img = std::vector<uint8_t>(tex->img->begin(), tex->img->end());
                images.push_back(img);
            } else {
                tex_id = fnd->second;
            }

            uint32_t tex_mask = TEXTURED_PARAM_MASK;
            SET_TEXTURE_ID(tex_mask, tex_id);
            m.base_color.r = *reinterpret_cast<float *>(&tex_mask);
        }

        m.metallic = mats[i].metallic;
        m.specular = mats[i].specular;
        m.roughness = mats[i].roughness;
        m.specular_tint = mats[i].specular_tint;
        m.anisotropy = mats[i].anisotropy;
        m.sheen = mats[i].sheen;
        m.sheen_tint = mats[i].sheen_tint;
        m.clearcoat = mats[i].clearcoat;
        m.clearcoat_gloss = mats[i].clearcoat_gloss;
        m.ior = mats[i].ior;
        m.specular_transmission = mats[i].specular_transmission;
        materials.push_back(m);
    }
}

Framebuffer::Framebuffer(const vec2i &size) : size(size.x, size.y), img(size.x * size.y, 0) {}
}
