#pragma once

#if __METAL_VERSION__

#include <metal_stdlib>

using namespace metal;

struct Geometry {
    device packed_float3 *vertices;
    device packed_uint3 *indices;
    device packed_float3 *normals;
    device packed_float2 *uvs;
    uint32_t num_normals;
    uint32_t num_uvs;
};

struct Instance {
    float4x4 inverse_transform;
    device uint32_t *geometries;
    device uint32_t *material_ids;
};

struct MaterialParams {
    packed_float3 base_color;
    float metallic;

    float specular;
    float roughness;
    float specular_tint;
    float anisotropy;

    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;

    float ior;
    float specular_transmission;
    float2 pad;
};

// Not sure if there's a cleaner way to pass a buffer of texture handles,
// Metal didn't like device texture2d<float> *textures as a buffer parameter
struct Texture {
    texture2d<float> tex;
};

#else
#include <Metal/Metal.h>
namespace params {

struct Geometry {
    uint64_t vertices;
    uint64_t indices;
    uint64_t normals;
    uint64_t uvs;
    uint32_t num_normals;
    uint32_t num_uvs;
};

struct Instance {
    glm::mat4 inverse_transform;
    uint64_t geometries;
    uint64_t material_ids;
};

struct MaterialParams {
    glm::vec3 base_color;
    float metallic;

    float specular;
    float roughness;
    float specular_tint;
    float anisotropy;

    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;

    float ior;
    float specular_transmission;
    glm::vec2 pad;
};

// Not sure if there's a cleaner way to pass a buffer of texture handles,
// Metal didn't like device texture2d<float> *textures as a buffer parameter
struct Texture {
    MTLResourceID tex;
};
}

#endif
