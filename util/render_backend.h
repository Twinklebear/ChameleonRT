#pragma once

#include <vector>
#include "scene.h"
#include <glm/glm.hpp>

struct RenderStats {
    float render_time = 0;
    float rays_per_second = 0;
};

struct RenderBackend {
    std::vector<uint32_t> img;

    virtual ~RenderBackend() {}

    virtual std::string name() = 0;

    virtual void initialize(const int fb_width, const int fb_height) = 0;

    // TODO Probably should take the scene through a shared_ptr
    virtual void set_scene(const Scene &scene) = 0;

    // Returns the rays per-second achieved, or -1 if this is not tracked
    virtual RenderStats render(const glm::vec3 &pos,
                               const glm::vec3 &dir,
                               const glm::vec3 &up,
                               const float fovy,
                               const bool camera_changed,
                               const bool readback_framebuffer) = 0;
};
