#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include "material.h"
#include "mesh.h"
#include <glm/glm.hpp>

#define CHECK_OPTIX(FN)                                                                \
    {                                                                                  \
        auto fn_err = FN;                                                              \
        if (fn_err != OPTIX_SUCCESS) {                                                 \
            std::cout << #FN << " failed due to " << optixGetErrorName(fn_err) << ": " \
                      << optixGetErrorString(fn_err) << std::endl                      \
                      << std::flush;                                                   \
            throw std::runtime_error(#FN);                                             \
        }                                                                              \
    }

#define CHECK_CUDA(FN)                                                                \
    {                                                                                 \
        auto fn_err = FN;                                                             \
        if (fn_err != cudaSuccess) {                                                  \
            std::cout << #FN << " failed due to " << cudaGetErrorName(fn_err) << ": " \
                      << cudaGetErrorString(fn_err) << std::endl                      \
                      << std::flush;                                                  \
            throw std::runtime_error(#FN);                                            \
        }                                                                             \
    }

namespace optix {

class Buffer {
    size_t buf_size = 0;
    void *ptr = nullptr;

public:
    Buffer() = default;
    Buffer(size_t size);
    ~Buffer();

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&b);
    Buffer &operator=(Buffer &&b);

    CUdeviceptr device_ptr() const;

    size_t size() const;

    void upload(const void *data, size_t size);

    void download(void *data, size_t size);

    template <typename T>
    void upload(const std::vector<T> &data);

    template <typename T, size_t N>
    void upload(const std::array<T, N> &data);

    template <typename T>
    void download(std::vector<T> &data);

    template <typename T, size_t N>
    void download(std::array<T, N> &data);

    void clear();
};

template <typename T>
void Buffer::upload(const std::vector<T> &data)
{
    upload(data.data(), data.size() * sizeof(T));
}

template <typename T, size_t N>
void Buffer::upload(const std::array<T, N> &data)
{
    upload(data.data(), data.size() * sizeof(T));
}

template <typename T>
void Buffer::download(std::vector<T> &data)
{
    download(data.data(), data.size() * sizeof(T));
}

template <typename T, size_t N>
void Buffer::download(std::array<T, N> &data)
{
    download(data.data(), data.size() * sizeof(T));
}

class Texture2D {
    glm::uvec2 tdims = glm::uvec2(0);
    cudaChannelFormatDesc channel_format;
    cudaArray_t data = 0;
    cudaTextureObject_t texture = 0;

public:
    Texture2D(glm::uvec2 dims, cudaChannelFormatDesc channel_format, ColorSpace color_space);
    ~Texture2D();

    Texture2D(Texture2D &&t);
    Texture2D &operator=(Texture2D &&t);

    Texture2D(const Texture2D &) = delete;
    Texture2D &operator=(const Texture2D &) = delete;

    void upload(const uint8_t *buf);

    cudaTextureObject_t handle();

    glm::uvec2 dims() const;
};

struct Geometry {
    std::shared_ptr<Buffer> vertex_buf, index_buf, normal_buf, uv_buf;
    uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    CUdeviceptr vertex_buf_ptr;

    Geometry() = default;

    // TODO: Allow other vertex and index formats? Right now this
    // assumes vec3f verts and uint3 indices
    Geometry(std::shared_ptr<Buffer> vertex_buf,
             std::shared_ptr<Buffer> index_buf,
             std::shared_ptr<Buffer> normal_buf,
             std::shared_ptr<Buffer> uv_buf,
             uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

    OptixBuildInput geom_desc() const;
};

class TriangleMesh {
    uint32_t build_flags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    std::vector<OptixBuildInput> build_inputs;

    Buffer build_output, scratch, post_build_info, bvh;

    OptixTraversableHandle as_handle;

public:
    std::vector<Geometry> geometries;

    TriangleMesh() = default;

    TriangleMesh(std::vector<Geometry> &geometries,
                 uint32_t build_flags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);

    // Enqueue the acceleration structure build construction into the passed stream
    void enqueue_build(OptixDeviceContext &device, CUstream &stream);

    // Enqueue the acceleration structure compaction into the passed stream
    void enqueue_compaction(OptixDeviceContext &device, CUstream &stream);

    // Finalize the BVH build structures to release any scratch space
    void finalize();

    OptixTraversableHandle handle();
};

class TopLevelBVH {
    OptixBuildInput geom_desc = {};

    uint32_t build_flags = OPTIX_BUILD_FLAG_NONE;

    Buffer build_output, scratch, post_build_info, bvh;

    OptixTraversableHandle as_handle;

public:
    std::shared_ptr<Buffer> instance_buf;
    std::vector<Instance> instances;

    TopLevelBVH() = default;

    TopLevelBVH(std::shared_ptr<Buffer> instance_buf,
                const std::vector<Instance> &instances,
                uint32_t build_flags = OPTIX_BUILD_FLAG_NONE);

    // Enqueue the acceleration structure build construction into the passed stream
    void enqueue_build(OptixDeviceContext &device, CUstream &stream);

    // Enqueue the acceleration structure compaction into the passed stream
    void enqueue_compaction(OptixDeviceContext &device, CUstream &stream);

    // Finalize the BVH build structures to release any scratch space
    void finalize();

    size_t num_instances() const;

    OptixTraversableHandle handle();
};

const static OptixModuleCompileOptions DEFAULT_MODULE_COMPILE_OPTIONS = {
    OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
    OPTIX_COMPILE_DEBUG_LEVEL_NONE,
};

class Module {
    OptixModule module;

    OptixProgramGroup create_program(OptixDeviceContext &device, OptixProgramGroupDesc &desc);

public:
    Module(OptixDeviceContext &device,
           const unsigned char *ptx,
           size_t ptex_len,
           const OptixModuleCompileOptions &compile_opts,
           const OptixPipelineCompileOptions &pipeline_opts);

    ~Module();

    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;

    OptixProgramGroup create_raygen(OptixDeviceContext &device, const std::string &function);

    OptixProgramGroup create_miss(OptixDeviceContext &device, const std::string &function);

    OptixProgramGroup create_hitgroup(OptixDeviceContext &device,
                                      const std::string &closest_hit,
                                      const std::string &any_hit = "",
                                      const std::string &intersection = "");
};

OptixPipeline compile_pipeline(OptixDeviceContext &device,
                               const OptixPipelineCompileOptions &compile_opts,
                               const OptixPipelineLinkOptions &link_opts,
                               const std::vector<OptixProgramGroup> &programs);

class ShaderTableBuilder;

struct ShaderRecord {
    std::string name;
    OptixProgramGroup program = {};
    size_t param_size = 0;

    ShaderRecord() = default;
    ShaderRecord(const std::string &name, OptixProgramGroup program, size_t param_size);
};

class ShaderTable {
    Buffer shader_table;
    std::vector<uint8_t> cpu_shader_table;

    OptixShaderBindingTable binding_table = {};

    std::unordered_map<std::string, size_t> record_offsets;

    ShaderTable(const ShaderRecord &raygen_record,
                const std::vector<ShaderRecord> &miss_records,
                const std::vector<ShaderRecord> &hitgroup_records);

    friend class ShaderTableBuilder;

public:
    ShaderTable() = default;

    /* Get the pointer to the start of the shader record, where the header
     * is written
     */
    uint8_t *get_shader_record(const std::string &shader);

    // Get a pointer to the parameters portion of the record for the shader
    template <typename T>
    T &get_shader_params(const std::string &shader);

    void upload();

    const OptixShaderBindingTable &table();
};

template <typename T>
T &ShaderTable::get_shader_params(const std::string &shader)
{
    return *reinterpret_cast<T *>(get_shader_record(shader) + OPTIX_SBT_RECORD_HEADER_SIZE);
}

class ShaderTableBuilder {
    ShaderRecord raygen_record;
    std::vector<ShaderRecord> miss_records;
    std::vector<ShaderRecord> hitgroup_records;

public:
    ShaderTableBuilder &set_raygen(const std::string &name,
                                   OptixProgramGroup program,
                                   size_t param_size);

    ShaderTableBuilder &add_miss(const std::string &name,
                                 OptixProgramGroup program,
                                 size_t param_size);

    ShaderTableBuilder &add_hitgroup(const std::string &name,
                                     OptixProgramGroup program,
                                     size_t param_size);

    ShaderTable build();
};

}
