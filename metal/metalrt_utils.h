#pragma once

#include <memory>
#include <Metal/Metal.h>
#include <simd/simd.h>
#include "mesh.h"

// TODO: Need to manually manage lifetimes, since I'm not
// sure if ARC will play very well with being used from a
// C++ class called from outside?
#if __has_feature(objc_arc)
#error "The Metal renderer uses manual reference counting"
#endif

namespace metal {

struct Context {
    id<MTLDevice> device = nullptr;
    id<MTLCommandQueue> command_queue = nullptr;

    Context();

    ~Context();

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

    ~ShaderLibrary();

    id<MTLFunction> new_function(NSString *name);
};

struct ComputePipeline {
    id<MTLComputePipelineState> pipeline = nullptr;

    ComputePipeline() = default;

    ComputePipeline(Context &context, id<MTLFunction> shader);

    ~ComputePipeline();
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

    ~Texture2D();

    const glm::uvec2 &dims() const;

    void get_bytes(void *out) const;

    size_t pixel_size() const;
};

struct Heap {
    id<MTLHeap> heap = nullptr;

    // Construct heaps using the HeapBuilder
    Heap() = default;

    ~Heap();
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

    ~Buffer();

    void *data();

    // Mark the entire buffer's contents as modified
    void mark_modified();

    // Mark a subrange of the buffer as modified
    void mark_range_modified(const glm::uvec2 range);

    size_t size() const;
};

struct HeapBuilder {
private:
    id<MTLDevice> device = nullptr;
    MTLHeapDescriptor *descriptor = nullptr;

public:
    HeapBuilder(Context &context);

    ~HeapBuilder();

    HeapBuilder &add_buffer(const size_t size, const MTLResourceOptions options);

    std::shared_ptr<Heap> build();
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

    BottomLevelBVH() = default;

    BottomLevelBVH(std::vector<Geometry> &geometries);

    void enqueue_build(Context &context,
                       id<MTLAccelerationStructureCommandEncoder> command_encoder);
};

struct TopLevelBVH : BVH {
    std::vector<Instance> instances;
    std::vector<std::shared_ptr<BottomLevelBVH>> meshes;

    std::shared_ptr<Buffer> instance_buffer;

    TopLevelBVH(std::vector<Instance> &instances,
                std::vector<std::shared_ptr<BottomLevelBVH>> &meshes);

    void enqueue_build(Context &context,
                       id<MTLAccelerationStructureCommandEncoder> command_encoder);
};

}

