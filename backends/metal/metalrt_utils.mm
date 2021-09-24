#include "metalrt_utils.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <Metal/Metal.h>
#include <simd/simd.h>
#include "util.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>

namespace metal {

Context::Context()
{
    // Find a Metal device that supports ray tracing
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    for (id<MTLDevice> d in devices) {
        if (d.supportsRaytracing && (!device || !d.isLowPower)) {
            device = d;
        }
    }
    if (!device) {
        std::cout << "No Metal device with ray tracing support found!\n";
        throw std::runtime_error("No Metal device with ray tracing support found");
    }

    command_queue = [device newCommandQueue];
}

std::string Context::device_name() const
{
    return [device.name UTF8String];
}

id<MTLCommandBuffer> Context::command_buffer()
{
    return [command_queue commandBuffer];
}

ShaderLibrary::ShaderLibrary(Context &context, const void *data, const size_t data_size)
{
    library_data = dispatch_data_create(data, data_size, nullptr, nullptr);
    NSError *err = nullptr;
    library = [context.device newLibraryWithData:library_data error:&err];

    if (!library) {
        std::cout << "Failed to load shader library: " << [err.localizedDescription UTF8String]
                  << "\n";
        throw std::runtime_error("Failed to load shader library");
    }
}

id<MTLFunction> ShaderLibrary::new_function(NSString *name)
{
    return [library newFunctionWithName:name];
}

ComputePipeline::ComputePipeline(Context &context, id<MTLFunction> shader)
{
    MTLComputePipelineDescriptor *pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = shader;
    pipeline_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = NO;

    NSError *err = nullptr;
    pipeline = [context.device newComputePipelineStateWithDescriptor:pipeline_desc
                                                             options:0
                                                          reflection:nil
                                                               error:&err];
    if (!pipeline) {
        std::cout << "Failed to create compute pipeline: " <<
            [err.localizedDescription UTF8String] << "\n";
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

MTLSize ComputePipeline::recommended_thread_group_size() const
{
    const size_t width = pipeline.threadExecutionWidth;
    const size_t height = pipeline.maxTotalThreadsPerThreadgroup / width;
    return MTLSizeMake(width, height, 1);
}

size_t Heap::size() const
{
    return heap.size;
}

Buffer::Buffer(Context &context, const size_t size, const MTLResourceOptions options)
    : options(options)
{
    buffer = [context.device newBufferWithLength:size options:options];
}

Buffer::Buffer(Heap &heap, const size_t size, const MTLResourceOptions options)
    : options(options)
{
    buffer = [heap.heap newBufferWithLength:size options:options];
}

void *Buffer::data()
{
    if (options & MTLResourceStorageModePrivate) {
        throw std::runtime_error("Cannot get pointer to CPU-inaccessible buffer");
    }
    return buffer.contents;
}

void Buffer::mark_modified()
{
    mark_range_modified(glm::uvec2(0, buffer.length));
}

void Buffer::mark_range_modified(const glm::uvec2 range)
{
    [buffer didModifyRange:NSMakeRange(range.x, range.y)];
}

size_t Buffer::size() const
{
    return buffer.length;
}

Texture2D::Texture2D(Context &context,
                     const uint32_t width,
                     const uint32_t height,
                     MTLPixelFormat format,
                     MTLTextureUsage usage)
    : tex_dims(width, height), format(format)
{
    MTLTextureDescriptor *tex_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    tex_desc.usage = usage;
    texture = [context.device newTextureWithDescriptor:tex_desc];
}

Texture2D::Texture2D(Heap &heap,
                     const uint32_t width,
                     const uint32_t height,
                     MTLPixelFormat format,
                     MTLTextureUsage usage)
    : tex_dims(width, height), format(format)
{
    MTLTextureDescriptor *tex_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    tex_desc.usage = usage;
    tex_desc.storageMode = MTLStorageModePrivate;
    texture = [heap.heap newTextureWithDescriptor:tex_desc];
}

const glm::uvec2 &Texture2D::dims() const
{
    return tex_dims;
}

void Texture2D::readback(void *out) const
{
    [texture getBytes:out
          bytesPerRow:tex_dims.x * pixel_size()
           fromRegion:MTLRegionMake2D(0, 0, tex_dims.x, tex_dims.y)
          mipmapLevel:0];
}

void Texture2D::upload(const void *data) const
{
    [texture replaceRegion:MTLRegionMake2D(0, 0, tex_dims.x, tex_dims.y)
               mipmapLevel:0
                 withBytes:data
               bytesPerRow:pixel_size() * tex_dims.x];
}

size_t Texture2D::pixel_size() const
{
    switch (format) {
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm:
        return 4;
    case MTLPixelFormatR16Uint:
        return 2;
    case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        throw std::runtime_error("Unhandled pixel format");
    }
}

HeapBuilder::HeapBuilder(Context &context) : device(context.device)
{
    descriptor = [MTLHeapDescriptor new];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = 0;
}

HeapBuilder &HeapBuilder::add_buffer(const size_t size, const MTLResourceOptions options)
{
    MTLSizeAndAlign size_align = [device heapBufferSizeAndAlignWithLength:size
                                                                  options:options];
    descriptor.size += align_to(size_align.size, size_align.align);

    return *this;
}

HeapBuilder &HeapBuilder::add_texture2d(const uint32_t width,
                                        const uint32_t height,
                                        MTLPixelFormat format,
                                        MTLTextureUsage usage)
{
    MTLTextureDescriptor *tex_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    tex_desc.usage = usage;
    tex_desc.storageMode = MTLStorageModePrivate;

    MTLSizeAndAlign size_align = [device heapTextureSizeAndAlignWithDescriptor:tex_desc];

    descriptor.size += align_to(size_align.size, size_align.align);

    return *this;
}

std::shared_ptr<Heap> HeapBuilder::build()
{
    std::shared_ptr<Heap> heap = std::make_shared<Heap>();
    heap->heap = [device newHeapWithDescriptor:descriptor];
    return heap;
}

void ArgumentEncoder::set_buffer(Buffer &buffer, const size_t offset, const size_t index)
{
    [encoder setBuffer:buffer.buffer offset:offset atIndex:index];
}

void ArgumentEncoder::set_texture(Texture2D &texture, const size_t index)
{
    [encoder setTexture:texture.texture atIndex:index];
}

void *ArgumentEncoder::constant_data_at(const size_t index)
{
    return [encoder constantDataAtIndex:index];
}

ArgumentEncoderBuilder::ArgumentEncoderBuilder(Context &context) : device(context.device) {}

ArgumentEncoderBuilder &ArgumentEncoderBuilder::add_buffer(const size_t index,
                                                           const MTLArgumentAccess access)
{
    MTLArgumentDescriptor *buf_desc = [MTLArgumentDescriptor argumentDescriptor];
    buf_desc.index = index;
    buf_desc.access = access;
    buf_desc.dataType = MTLDataTypePointer;
    [arguments addObject:buf_desc];

    return *this;
}

ArgumentEncoderBuilder &ArgumentEncoderBuilder::add_texture(const size_t index,
                                                            const MTLArgumentAccess access)
{
    MTLArgumentDescriptor *buf_desc = [MTLArgumentDescriptor argumentDescriptor];
    buf_desc.index = index;
    buf_desc.access = access;
    buf_desc.dataType = MTLDataTypeTexture;
    [arguments addObject:buf_desc];

    return *this;
}

ArgumentEncoderBuilder &ArgumentEncoderBuilder::add_constant(const size_t index,
                                                             const MTLDataType type)
{
    MTLArgumentDescriptor *buf_desc = [MTLArgumentDescriptor argumentDescriptor];
    buf_desc.index = index;
    buf_desc.access = MTLArgumentAccessReadOnly;
    buf_desc.dataType = type;
    [arguments addObject:buf_desc];

    return *this;
}

size_t ArgumentEncoderBuilder::encoded_length() const
{
    id<MTLArgumentEncoder> encoder = [device newArgumentEncoderWithArguments:arguments];
    const size_t length = encoder.encodedLength;
    return length;
}

std::shared_ptr<ArgumentEncoder> ArgumentEncoderBuilder::encoder_for_buffer(
    Buffer &buffer, const size_t offset)
{
    auto encoder = std::make_shared<ArgumentEncoder>();
    encoder->encoder = [device newArgumentEncoderWithArguments:arguments];
    [encoder->encoder setArgumentBuffer:buffer.buffer offset:offset];
    return encoder;
}

Geometry::Geometry(const std::shared_ptr<Buffer> &vertex_buf,
                   const std::shared_ptr<Buffer> &index_buf,
                   const std::shared_ptr<Buffer> &normal_buf,
                   const std::shared_ptr<Buffer> &uv_buf)
    : vertex_buf(vertex_buf), index_buf(index_buf), normal_buf(normal_buf), uv_buf(uv_buf)
{
}

size_t Geometry::num_tris() const
{
    return index_buf->size() / sizeof(glm::uvec3);
}

void BVH::build(Context &context,
                id<MTLAccelerationStructureCommandEncoder> command_encoder,
                MTLAccelerationStructureDescriptor *desc)
{
    MTLAccelerationStructureSizes accel_sizes =
        [context.device accelerationStructureSizesWithDescriptor:desc];

    bvh = [context.device
        newAccelerationStructureWithSize:accel_sizes.accelerationStructureSize];

    scratch_buffer = std::make_shared<Buffer>(
        context, accel_sizes.buildScratchBufferSize, MTLResourceStorageModePrivate);

    compacted_size_buffer =
        std::make_shared<Buffer>(context, sizeof(uint32_t), MTLResourceStorageModeShared);

    // Queue the build
    [command_encoder buildAccelerationStructure:bvh
                                     descriptor:desc
                                  scratchBuffer:scratch_buffer->buffer
                            scratchBufferOffset:0];

    // Get the compacted size back from the build
    [command_encoder writeCompactedAccelerationStructureSize:bvh
                                                    toBuffer:compacted_size_buffer->buffer
                                                      offset:0];
}

void BVH::enqueue_compaction(Context &context,
                             id<MTLAccelerationStructureCommandEncoder> command_encoder)
{
    const uint32_t compact_size = *reinterpret_cast<uint32_t *>(compacted_size_buffer->data());
    id<MTLAccelerationStructure> compact_as =
        [context.device newAccelerationStructureWithSize:compact_size];
    [command_encoder copyAndCompactAccelerationStructure:bvh
                                 toAccelerationStructure:compact_as];

    // Cleanup temporary build buffers and non-compact BVH
    compacted_size_buffer = nullptr;
    scratch_buffer = nullptr;

    bvh = compact_as;
}

BottomLevelBVH::BottomLevelBVH(const std::vector<Geometry> &geometries,
                               std::shared_ptr<Buffer> &geometry_id_buffer)
    : geometries(geometries), geometry_id_buffer(geometry_id_buffer)
{
}

void BottomLevelBVH::enqueue_build(Context &context,
                                   id<MTLAccelerationStructureCommandEncoder> command_encoder)
{
    NSMutableArray *geom_descs = [NSMutableArray array];

    for (size_t i = 0; i < geometries.size(); ++i) {
        const auto &g = geometries[i];
        MTLAccelerationStructureTriangleGeometryDescriptor *g_desc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

        g_desc.vertexBuffer = g.vertex_buf->buffer;
        g_desc.vertexStride = sizeof(glm::vec3);
        g_desc.triangleCount = g.num_tris();

        g_desc.indexBuffer = g.index_buf->buffer;
        g_desc.indexType = MTLIndexTypeUInt32;

        g_desc.intersectionFunctionTableOffset = i;
        g_desc.opaque = YES;

        [geom_descs addObject:g_desc];
    }

    MTLPrimitiveAccelerationStructureDescriptor *bvh_desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    bvh_desc.geometryDescriptors = geom_descs;

    build(context, command_encoder, bvh_desc);
}

TopLevelBVH::TopLevelBVH(const std::vector<ParameterizedMesh> &parameterized_meshes,
                         const std::vector<Instance> &instances,
                         std::vector<std::shared_ptr<BottomLevelBVH>> &meshes)
    : parameterized_meshes(parameterized_meshes), instances(instances), meshes(meshes)
{
}

void TopLevelBVH::enqueue_build(Context &context,
                                id<MTLAccelerationStructureCommandEncoder> command_encoder)
{
    NSMutableArray *blas_array = [NSMutableArray array];
    for (const auto &blas : meshes) {
        [blas_array addObject:blas->bvh];
    }

    std::vector<uint32_t> parameterized_mesh_sbt_offsets;
    {
        // Compute the offsets each parameterized mesh will be written too in the SBT,
        // these are then the instance SBT offsets shared by each instance
        uint32_t offset = 0;
        for (const auto &pm : parameterized_meshes) {
            parameterized_mesh_sbt_offsets.push_back(offset);
            offset += meshes[pm.mesh_id]->geometries.size();
        }
    }

    instance_buffer = std::make_shared<Buffer>(
        context,
        instances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor),
        MTLResourceStorageModeManaged);

    MTLAccelerationStructureInstanceDescriptor *inst_descs =
        reinterpret_cast<MTLAccelerationStructureInstanceDescriptor *>(
            instance_buffer->data());

    for (size_t i = 0; i < instances.size(); ++i) {
        const auto &inst = instances[i];

        inst_descs[i].accelerationStructureIndex =
            parameterized_meshes[inst.parameterized_mesh_id].mesh_id;
        inst_descs[i].intersectionFunctionTableOffset =
            parameterized_mesh_sbt_offsets[inst.parameterized_mesh_id];
        inst_descs[i].mask = 0xff;

        const glm::mat4x3 tfm = inst.transform;
        std::memcpy(&inst_descs[i].transformationMatrix,
                    glm::value_ptr(tfm),
                    sizeof(MTLPackedFloat4x3));
    }
    instance_buffer->mark_modified();

    MTLInstanceAccelerationStructureDescriptor *tlas_desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];

    tlas_desc.instancedAccelerationStructures = blas_array;
    tlas_desc.instanceDescriptorBuffer = instance_buffer->buffer;
    tlas_desc.instanceCount = instances.size();
    tlas_desc.instanceDescriptorBufferOffset = 0;
    tlas_desc.instanceDescriptorStride = sizeof(MTLAccelerationStructureInstanceDescriptor);

    build(context, command_encoder, tlas_desc);
}

};

