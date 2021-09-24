#pragma once

#include <ospray/ospray.h>
#include <ospray/ospray_util.h>
#include "render_backend.h"

struct RenderOSPRay : RenderBackend {
    OSPCamera camera;
    OSPRenderer renderer;
    OSPFrameBuffer fb;
    OSPWorld world;

    Scene scene;
    std::vector<OSPTexture> textures;
    std::vector<OSPMaterial> materials;
    std::vector<OSPInstance> instances;
    std::vector<OSPLight> lights;

    RenderOSPRay();
    ~RenderOSPRay();

    std::string name() override;
    void initialize(const int fb_width, const int fb_height) override;
    void set_scene(const Scene &scene) override;
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed,
                       const bool need_readback) override;

private:
    void set_material_param(OSPMaterial &mat, const std::string &name, const float val) const;
};
