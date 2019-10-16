#pragma once

#include <ospray/ospray.h>
#include "render_backend.h"

struct RenderOSPRay : RenderBackend {
    OSPCamera camera;
    OSPRenderer renderer;
    OSPFrameBuffer fb;
    OSPWorld world;

    Scene scene;
    std::vector<OSPTexture> textures;
    std::vector<OSPMaterial> materials;
    std::vector<OSPGroup> meshes;
    std::vector<OSPInstance> instances;
    std::vector<OSPLight> lights;

    RenderOSPRay();

    std::string name() override;
    void initialize(const int fb_width, const int fb_height) override;
    void set_scene(const Scene &scene) override;
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed) override;
};
