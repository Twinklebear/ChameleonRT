#pragma once

#include "float3.h"

namespace kernel {

// Column major 4x4 matrix to match GLM
// TODO: swap to 3x4 row-major to match other backends
struct mat4 {
    float m[16];
};

void load_mat4(mat4 &m, const float *buf)
{
    for (uint32_t i = 0; i < 16; ++i) {
        m.m[i] = buf[i];
    }
}

void transpose(mat4 &m)
{
    for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t i = j + 1; i < 4; ++i) {
            const float x = m.m[j * 4 + i];
            m.m[j * 4 + i] = m.m[i * 4 + j];
            m.m[i * 4 + j] = x;
        }
    }
}

float3 mul(const mat4 &m, const float3 &v)
{
    float3 res = make_float3(0.f);
    res.x = m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z;
    res.y = m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z;
    res.z = m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z;
    return res;
}

}
