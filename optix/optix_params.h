#pragma once

struct MaterialParams {
#ifdef __CUDA_ARCH__
    float3 base_color;
#else
    glm::vec3 base_color;
#endif
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
    float pad1, pad2;
};

struct LaunchParams {
#ifdef __CUDA_ARCH__
    float4 cam_pos;
    float4 cam_du;
    float4 cam_dv;
    float4 cam_dir_top_left;
#else
    glm::vec4 cam_pos;
    glm::vec4 cam_du;
    glm::vec4 cam_dv;
    glm::vec4 cam_dir_top_left;
#endif

    uint32_t frame_id;

#ifdef __CUDA_ARCH__
    uchar4 *framebuffer;
    float4 *accum_buffer;
    cudaTextureObject_t *textures;
#else
    CUdeviceptr framebuffer;
    CUdeviceptr accum_buffer;
    CUdeviceptr textures;
#endif

#ifdef REPORT_RAY_STATS
#ifdef __CUDA_ARCH__
    uint16_t *ray_stats_buffer;
#else
    CUdeviceptr ray_stats_buffer;
#endif
#endif

    OptixTraversableHandle scene;
};

struct RayGenParams {
#ifdef __CUDA_ARCH__
    MaterialParams *materials;
    QuadLight *lights;
#else
    CUdeviceptr materials;
    CUdeviceptr lights;
#endif
    uint32_t num_lights;
};

struct HitGroupParams {
#ifdef __CUDA_ARCH__
    float3 *vertex_buffer;
    uint3 *index_buffer;
    float2 *uv_buffer;
    float3 *normal_buffer;
    uint32_t material_id;
#else
    CUdeviceptr vertex_buffer;
    CUdeviceptr index_buffer;
    CUdeviceptr uv_buffer;
    CUdeviceptr normal_buffer;
    uint32_t material_id;
#endif
};
