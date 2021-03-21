#pragma once

#include <memory>
#include <Metal/Metal.h>
#include <simd/simd.h>
#include "mesh.h"

namespace metal {

struct Context {
    id<MTLDevice> device = nullptr;
    id<MTLCommandQueue> command_queue = nullptr;

    Context();

    std::string device_name() const;

    id<MTLCommandBuffer> command_buffer();
};

struct ShaderLibrary {
private:
    dispatch_data_t library_data;

public:
    id<MTLLibrary> library = nullptr;

    ShaderLibrary() = default;

    ShaderLibrary(Context &context, const void *data, const size_t data_size);

    id<MTLFunction> new_function(NSString *name);
};

struct ComputePipeline {
    id<MTLComputePipelineState> pipeline = nullptr;

    ComputePipeline() = default;

    ComputePipeline(Context &context, id<MTLFunction> shader);

    MTLSize recommended_thread_group_size() const;
};

struct Heap {
    id<MTLHeap> heap = nullptr;

    // Construct heaps using the HeapBuilder
    Heap() = default;

    size_t size() const;
};

struct Buffer {
private:
    MTLResourceOptions options;

public:
    id<MTLBuffer> buffer = nullptr;

    Buffer() = default;

    Buffer(Context &context, const size_t size, const MTLResourceOptions options);

    // Allocate the buffer from the passed heap
    Buffer(Heap &heap, const size_t size, const MTLResourceOptions options);

    void *data();

    // Mark the entire buffer's contents as modified
    void mark_modified();

    // Mark a subrange of the buffer as modified
    void mark_range_modified(const glm::uvec2 range);

    size_t size() const;
};

struct Texture2D {
private:
    glm::uvec2 tex_dims;
    MTLPixelFormat format;

public:
    id<MTLTexture> texture = nullptr;

    Texture2D() = default;

    Texture2D(Context &context,
              const uint32_t width,
              const uint32_t height,
              MTLPixelFormat format,
              MTLTextureUsage usage);

    // Allocate the texture from the passed heap
    Texture2D(Heap &heap,
              const uint32_t width,
              const uint32_t height,
              MTLPixelFormat format,
              MTLTextureUsage usage);

    const glm::uvec2 &dims() const;

    void readback(void *out) const;

    void upload(const void *data) const;

    size_t pixel_size() const;
};

struct HeapBuilder {
private:
    id<MTLDevice> device = nullptr;
    MTLHeapDescriptor *descriptor = nullptr;

public:
    HeapBuilder(Context &context);

    HeapBuilder &add_buffer(const size_t size, const MTLResourceOptions options);

    HeapBuilder &add_texture2d(const uint32_t width,
                               const uint32_t height,
                               MTLPixelFormat format,
                               MTLTextureUsage usage);

    std::shared_ptr<Heap> build();
};

struct ArgumentEncoder {
    id<MTLArgumentEncoder> encoder = nullptr;

    ArgumentEncoder() = default;

    void set_buffer(Buffer &buffer, const size_t offset, const size_t index);

    void set_texture(Texture2D &texture, const size_t index);

    // TODO: Could do some template here and type validation
    void *constant_data_at(const size_t index);
};

struct ArgumentEncoderBuilder {
private:
    id<MTLDevice> device = nullptr;
    NSMutableArray *arguments = [NSMutableArray array];

public:
    ArgumentEncoderBuilder(Context &context);

    ArgumentEncoderBuilder &add_buffer(const size_t index, const MTLArgumentAccess access);

    ArgumentEncoderBuilder &add_texture(const size_t index, const MTLArgumentAccess access);

    ArgumentEncoderBuilder &add_constant(const size_t index, const MTLDataType type);

    size_t encoded_length() const;

    // Note: it doesn't seem like you can re-use the same argument encoder but swap out
    // the buffer it's writing too, so the encoder is not very reusable
    std::shared_ptr<ArgumentEncoder> encoder_for_buffer(Buffer &buffer, const size_t offset);
};

struct Geometry {
    std::shared_ptr<Buffer> vertex_buf, index_buf, normal_buf, uv_buf;

    Geometry() = default;

    // TODO later take geometry flags
    Geometry(const std::shared_ptr<Buffer> &vertex_buf,
             const std::shared_ptr<Buffer> &index_buf,
             const std::shared_ptr<Buffer> &normal_buf,
             const std::shared_ptr<Buffer> &uv_buf);

    size_t num_tris() const;
};

struct BVH {
protected:
    std::shared_ptr<Buffer> compacted_size_buffer;
    std::shared_ptr<Buffer> scratch_buffer;

    void build(Context &context,
               id<MTLAccelerationStructureCommandEncoder> command_encoder,
               MTLAccelerationStructureDescriptor *desc);

public:
    id<MTLAccelerationStructure> bvh;

    virtual ~BVH() = default;

    virtual void enqueue_build(Context &context,
                               id<MTLAccelerationStructureCommandEncoder> command_encoder) = 0;

    void enqueue_compaction(Context &context,
                            id<MTLAccelerationStructureCommandEncoder> command_encoder);
};

struct BottomLevelBVH : BVH {
    std::vector<Geometry> geometries;
    // The indices of the geometry of this BLAS in the global geometry list
    std::shared_ptr<Buffer> geometry_id_buffer;

    BottomLevelBVH() = default;

    BottomLevelBVH(const std::vector<Geometry> &geometries,
                   std::shared_ptr<Buffer> &geometry_id_buffer);

    void enqueue_build(Context &context,
                       id<MTLAccelerationStructureCommandEncoder> command_encoder);
};

struct TopLevelBVH : BVH {
    std::vector<ParameterizedMesh> parameterized_meshes;
    std::vector<Instance> instances;
    std::vector<std::shared_ptr<BottomLevelBVH>> meshes;

    std::shared_ptr<Buffer> instance_buffer;

    TopLevelBVH(const std::vector<ParameterizedMesh> &parameterized_meshes,
                const std::vector<Instance> &instances,
                std::vector<std::shared_ptr<BottomLevelBVH>> &meshes);

    void enqueue_build(Context &context,
                       id<MTLAccelerationStructureCommandEncoder> command_encoder);
};

}

