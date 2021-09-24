#pragma once

#include <simd/simd.h>

struct ViewParams {
    simd::float4 cam_pos;
    simd::float4 cam_du;
    simd::float4 cam_dv;
    simd::float4 cam_dir_top_left;
    simd::uint2 fb_dims;
    uint32_t frame_id;
    uint32_t num_lights;
};

