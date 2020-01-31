#pragma once

#include <vector>
#include <ospcommon/math/AffineSpace.h>
#include <ospcommon/math/vec.h>
#include <ospcommon/utility/ParameterizedObject.h>
#include "data.h"
#include "scene.h"
#include <glm/glm.hpp>

namespace device {

class APIObject : public ospcommon::utility::ParameterizedObject {
public:
    virtual ~APIObject() {}

    virtual void commit(){};
};

class Light : public APIObject {
public:
    QuadLight light;

    void commit() override;
};

class Camera : public APIObject {
public:
    ::Camera camera;

    void commit() override;
};

class Geometry : public APIObject {
public:
    Data *vertices = nullptr;
    Data *normals = nullptr;
    Data *uvs = nullptr;
    Data *indices = nullptr;

    void commit() override;
};

class GeometricModel : public APIObject {
public:
    Geometry *geometry = nullptr;
    uint32_t material_id = -1;

    GeometricModel(Geometry *geom);
    GeometricModel() = default;

    void commit() override;
};

class Group : public APIObject {
public:
    Data *geometry = nullptr;

    void commit() override;
};

class Texture : public APIObject {
public:
    int format = -1;
    int channels = 0;
    Data *img = nullptr;

    void commit() override;
};

class Material : public APIObject {
public:
    glm::vec3 base_color = glm::vec3(0.9f);
    Texture *tex_base_color = nullptr;

    float metallic = 0;
    float specular = 0;
    float roughness = 1;
    float specular_tint = 0;
    float anisotropy = 0;
    float sheen = 0;
    float sheen_tint = 0;
    float clearcoat = 0;
    float clearcoat_gloss = 0;
    float ior = 1.5;
    float specular_transmission = 0;

    void commit() override;
};

class Instance : public APIObject {
public:
    Group *group = nullptr;
    glm::mat4 transform;

    Instance(Group *group);
    Instance() = default;

    void commit() override;
};

// TODO: For a more final mapping of RTX/SBT-based backends, the world would
// store the SBT, and the renderer would be the ray gen and miss programs
class World : public APIObject {
public:
    Scene scene;

    void commit() override;
};

class Renderer : public APIObject {
public:
    World *last_world = nullptr;
    std::vector<DisneyMaterial> materials;
    std::vector<Image> images;

    void commit() override;
};

class Framebuffer : public APIObject {
public:
    glm::ivec2 size;
    std::vector<uint32_t> img;

    Framebuffer(const ospcommon::math::vec2i &size);
    Framebuffer() = default;
};
}
