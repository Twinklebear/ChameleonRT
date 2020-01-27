#pragma once

#include <string>
#include <vector>

struct Display {
    virtual ~Display() {}

    virtual std::string gpu_brand() = 0;

    virtual std::string name() = 0;

    virtual void resize(const int fb_width, const int fb_height) = 0;

    virtual void new_frame() = 0;

    virtual void display(const std::vector<uint32_t> &img) = 0;
};
