#include "render_metal.h"
#include <iostream>
#include <Cocoa/Cocoa.h>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.h>
#include "render_metal_embedded_metallib.h"
#include "shader_types.h"
#include "util.h"
#include <glm/ext.hpp>

// TODO: Need to manually manage lifetimes, since I'm not
// sure if ARC will play very well with being used from a
// C++ class called from outside?
#if __has_feature(objc_arc)
#error "Do not enable ARC"
#endif

struct RenderMetalData {
    id<MTLDevice> device = nullptr;
    id<MTLCommandQueue> command_queue = nullptr;

    dispatch_data_t shader_library_data;
    id<MTLLibrary> shader_library = nullptr;

    id<MTLTexture> render_target = nullptr;

    id<MTLBuffer> vertex_buffer = nullptr;
    id<MTLBuffer> index_buffer = nullptr;
    id<MTLBuffer> instance_buffer = nullptr;

    id<MTLAccelerationStructure> blas = nullptr;
    id<MTLAccelerationStructure> tlas = nullptr;

    id<MTLComputePipelineState> pipeline = nullptr;

    id<MTLBuffer> geom_arg_buffer = nullptr;

    // Heap containing all the geometry's vertex and index buffers
    id<MTLHeap> geometry_heap = nullptr;

    // TODO: Destructor to clean up, we're not using ARC
};

// TODO: Decide on and implement abstractions, will follow similar
// to the other backends
id<MTLAccelerationStructure> build_acceleration_structure(
    id<MTLDevice> device,
    id<MTLCommandQueue> command_queue,
    MTLAccelerationStructureDescriptor *desc);

RenderMetal::RenderMetal()
{
    metal = std::make_shared<RenderMetalData>();

    // Find a Metal device that supports ray tracing
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    for (id<MTLDevice> d in devices) {
        if (d.supportsRaytracing && (!metal->device || !d.isLowPower)) {
            metal->device = d;
        }
    }
    // TODO: check/throw if no device, means we don't have ray tracing support

    std::cout << "Selected Metal device " << [metal->device.name UTF8String] << "\n";

    metal->command_queue = [metal->device newCommandQueue];

    metal->shader_library_data = dispatch_data_create(
        render_metal_metallib, sizeof(render_metal_metallib), nullptr, nullptr);

    NSError *err = nullptr;
    metal->shader_library = [metal->device newLibraryWithData:metal->shader_library_data
                                                        error:&err];
    if (!metal->shader_library) {
        std::cout << "Failed to load shader library: " << [err.localizedDescription UTF8String]
                  << "\n";
    }
}

RenderMetal::~RenderMetal()
{
    // TODO: Cleanup
}

std::string RenderMetal::name()
{
    return "Metal Ray Tracing";
}

void RenderMetal::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    fb_dims = glm::uvec2(fb_width, fb_height);
    img.resize(fb_width * fb_height);

    // Make our render target (TODO: also need a float accum target)
    MTLTextureDescriptor *tex_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:fb_width
                                                          height:fb_height
                                                       mipmapped:NO];
    tex_desc.usage = MTLTextureUsageShaderWrite;
    // TODO: Explciitly release render target
    // TODO: Later render target
    metal->render_target = [metal->device newTextureWithDescriptor:tex_desc];
}

void RenderMetal::set_scene(const Scene &scene)
{
    // TODO Testing: Just take the first mesh of the first instance for now
    const Geometry &geom = scene.meshes[scene.instances[0].mesh_id].geometries[0];

    // Create a heap to hold all the geometry buffers
    {
        MTLHeapDescriptor *heap_desc = [MTLHeapDescriptor new];
        heap_desc.storageMode = MTLStorageModePrivate;
        heap_desc.size = 0;

        // Compute the size needed to hold everything in the heap
        MTLSizeAndAlign size_align = [metal->device
            heapBufferSizeAndAlignWithLength:sizeof(glm::vec3) * geom.vertices.size()
                                     options:MTLResourceStorageModePrivate];
        heap_desc.size += align_to(size_align.size, size_align.align);

        size_align = [metal->device
            heapBufferSizeAndAlignWithLength:sizeof(glm::uvec3) * geom.indices.size()
                                     options:MTLResourceStorageModePrivate];
        heap_desc.size += align_to(size_align.size, size_align.align);

        metal->geometry_heap = [metal->device newHeapWithDescriptor:heap_desc];
    }

    // Upload the data to staging and copy it into the heap
    {
        // TODO: Released automatically right?
        id<MTLBuffer> vertex_buffer =
            [metal->device newBufferWithLength:sizeof(glm::vec3) * geom.vertices.size()
                                       options:MTLResourceStorageModeManaged];
        std::memcpy(vertex_buffer.contents, geom.vertices.data(), vertex_buffer.length);
        [vertex_buffer didModifyRange:NSMakeRange(0, vertex_buffer.length)];

        id<MTLBuffer> index_buffer =
            [metal->device newBufferWithLength:sizeof(glm::uvec3) * geom.indices.size()
                                       options:MTLResourceStorageModeManaged];
        std::memcpy(index_buffer.contents, geom.indices.data(), index_buffer.length);
        [index_buffer didModifyRange:NSMakeRange(0, index_buffer.length)];

        // Allocate the buffers from the heap and copy the data into them
        metal->vertex_buffer =
            [metal->geometry_heap newBufferWithLength:vertex_buffer.length
                                              options:MTLResourceStorageModePrivate];

        metal->index_buffer =
            [metal->geometry_heap newBufferWithLength:index_buffer.length
                                              options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> command_buffer = [metal->command_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit_encoder = command_buffer.blitCommandEncoder;

        [blit_encoder copyFromBuffer:vertex_buffer
                        sourceOffset:0
                            toBuffer:metal->vertex_buffer
                   destinationOffset:0
                                size:metal->vertex_buffer.length];

        [blit_encoder copyFromBuffer:index_buffer
                        sourceOffset:0
                            toBuffer:metal->index_buffer
                   destinationOffset:0
                                size:metal->index_buffer.length];

        [blit_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }

    // Build argument buffer for the mesh
    id<MTLArgumentEncoder> argument_encoder;
    {
        MTLArgumentDescriptor *vertex_buf_desc = [MTLArgumentDescriptor argumentDescriptor];
        vertex_buf_desc.index = 0;
        vertex_buf_desc.access = MTLArgumentAccessReadOnly;
        vertex_buf_desc.dataType = MTLDataTypePointer;

        MTLArgumentDescriptor *index_buf_desc = [MTLArgumentDescriptor argumentDescriptor];
        index_buf_desc.index = 1;
        index_buf_desc.access = MTLArgumentAccessReadOnly;
        index_buf_desc.dataType = MTLDataTypePointer;

        argument_encoder =
            [metal->device newArgumentEncoderWithArguments:@[vertex_buf_desc, index_buf_desc]];
    }

    // This will be the stride between consecutive geometries, though right now
    // we just have one so it's also the buffer size we need
    const uint32_t geom_args_stride = argument_encoder.encodedLength;
    std::cout << "Geom args length: " << geom_args_stride << "b\n";
    metal->geom_arg_buffer = [metal->device newBufferWithLength:geom_args_stride
                                                        options:MTLResourceStorageModeManaged];
    // Write the arguments into the buffer
    [argument_encoder setArgumentBuffer:metal->geom_arg_buffer offset:0];
    [argument_encoder setBuffer:metal->vertex_buffer offset:0 atIndex:0];
    [argument_encoder setBuffer:metal->index_buffer offset:0 atIndex:1];

    [metal->geom_arg_buffer didModifyRange:NSMakeRange(0, metal->geom_arg_buffer.length)];

    // Setup the geometry descriptor for this triangle geometry
    MTLAccelerationStructureTriangleGeometryDescriptor *geom_desc =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geom_desc.vertexBuffer = metal->vertex_buffer;
    geom_desc.vertexStride = sizeof(glm::vec3);
    geom_desc.triangleCount = geom.num_tris();

    geom_desc.indexBuffer = metal->index_buffer;
    geom_desc.indexType = MTLIndexTypeUInt32;

    // TODO: Seems like Metal is inline ray tracing style, but also has some stuff
    // for "opaque triangle" intersection functions and visible ones? How does the
    // pipeline map out when using those? Are they more like any hit or closest hit?
    // Won't be using intersection table here for now
    geom_desc.intersectionFunctionTableOffset = 0;
    geom_desc.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor *blas_desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    blas_desc.geometryDescriptors = @[geom_desc];
    metal->blas = build_acceleration_structure(metal->device, metal->command_queue, blas_desc);

    // Setup the instances for the TLAS
    metal->instance_buffer =
        [metal->device newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor)
                                   options:MTLResourceStorageModeManaged];

    {
        MTLAccelerationStructureInstanceDescriptor *instance =
            reinterpret_cast<MTLAccelerationStructureInstanceDescriptor *>(
                metal->instance_buffer.contents);

        instance->accelerationStructureIndex = 0;
        instance->intersectionFunctionTableOffset = 0;
        instance->mask = 1;

        // Note: Column-major in Metal
        std::memset(&instance->transformationMatrix, 0, sizeof(MTLPackedFloat4x3));
        instance->transformationMatrix.columns[0][0] = 1.f;
        instance->transformationMatrix.columns[1][1] = 1.f;
        instance->transformationMatrix.columns[2][2] = 1.f;

        [metal->instance_buffer didModifyRange:NSMakeRange(0, metal->instance_buffer.length)];
    }

    // Now build the TLAS
    MTLInstanceAccelerationStructureDescriptor *tlas_desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlas_desc.instancedAccelerationStructures = @[metal->blas];
    tlas_desc.instanceDescriptorBuffer = metal->instance_buffer;
    tlas_desc.instanceCount = 1;
    tlas_desc.instanceDescriptorBufferOffset = 0;
    tlas_desc.instanceDescriptorStride = sizeof(MTLAccelerationStructureInstanceDescriptor);

    metal->tlas = build_acceleration_structure(metal->device, metal->command_queue, tlas_desc);

    // Setup the compute pipeline
    id<MTLFunction> raygen_shader = [metal->shader_library newFunctionWithName:@"raygen"];
    MTLComputePipelineDescriptor *pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = raygen_shader;
    // TODO Later: make this yes and have better threadgroup setup
    pipeline_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = NO;

    NSError *err = nullptr;
    metal->pipeline = [metal->device newComputePipelineStateWithDescriptor:pipeline_desc
                                                                   options:0
                                                                reflection:nil
                                                                     error:&err];
    if (!metal->pipeline) {
        std::cout << "Failed to create compute pipeline: " <<
            [err.localizedDescription UTF8String] << "\n";
    }
}

RenderStats RenderMetal::render(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy,
                                const bool camera_changed,
                                const bool readback_framebuffer)
{
    RenderStats stats;

    if (camera_changed) {
        frame_id = 0;
    }

    ViewParams view_params = compute_view_parameters(pos, dir, up, fovy);

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [metal->command_queue commandBuffer];

        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setTexture:metal->render_target atIndex:0];

        // Embed the view params in the command buffer
        // TODO: It seems like this is the best way to pass small constants that potentially
        // change every frame?
        [command_encoder setBytes:&view_params length:sizeof(ViewParams) atIndex:0];

        [command_encoder setAccelerationStructure:metal->tlas atBufferIndex:1];
        [command_encoder useResource:metal->blas usage:MTLResourceUsageRead];

        [command_encoder setBuffer:metal->geom_arg_buffer offset:0 atIndex:2];
        [command_encoder useHeap:metal->geometry_heap];

        [command_encoder setBuffer:metal->instance_buffer offset:0 atIndex:3];

        [command_encoder setComputePipelineState:metal->pipeline];
        // TODO: Better thread group sizing here, this is a poor choice for utilization
        // but keeps the example simple
        [command_encoder dispatchThreadgroups:MTLSizeMake(fb_dims.x, fb_dims.y, 1)
                        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }

    if (readback_framebuffer || !native_display) {
        [metal->render_target getBytes:img.data()
                           bytesPerRow:fb_dims.x * sizeof(uint32_t)
                            fromRegion:MTLRegionMake2D(0, 0, fb_dims.x, fb_dims.y)
                           mipmapLevel:0];
    }

    ++frame_id;
    return stats;
}

ViewParams RenderMetal::compute_view_parameters(const glm::vec3 &pos,
                                                const glm::vec3 &dir,
                                                const glm::vec3 &up,
                                                const float fovy)
{
    glm::vec2 img_plane_size;
    img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
    img_plane_size.x = img_plane_size.y * static_cast<float>(fb_dims.x) / fb_dims.y;

    const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
    const glm::vec3 dir_dv = -glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
    const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

    ViewParams view_params;
    view_params.cam_pos = simd::float4{pos.x, pos.y, pos.z, 1.f};
    view_params.cam_du = simd::float4{dir_du.x, dir_du.y, dir_du.z, 0.f};
    view_params.cam_dv = simd::float4{dir_dv.x, dir_dv.y, dir_dv.z, 0.f};
    view_params.cam_dir_top_left =
        simd::float4{dir_top_left.x, dir_top_left.y, dir_top_left.z, 0.f};
    view_params.fb_dims = simd::uint2{fb_dims.x, fb_dims.y};
    view_params.frame_id = frame_id;

    return view_params;
}

id<MTLAccelerationStructure> build_acceleration_structure(
    id<MTLDevice> device,
    id<MTLCommandQueue> command_queue,
    MTLAccelerationStructureDescriptor *desc)
{
    // Does it need an autoreleaseblock?
    // Build then compact the acceleration structure
    MTLAccelerationStructureSizes accel_sizes =
        [device accelerationStructureSizesWithDescriptor:desc];
    std::cout << "Acceleration structure sizes:\n"
              << "\tstructure size: " << accel_sizes.accelerationStructureSize << "b\n"
              << "\tscratch size: " << accel_sizes.buildScratchBufferSize << "b\n";

    id<MTLAccelerationStructure> scratch_as =
        [device newAccelerationStructureWithSize:accel_sizes.accelerationStructureSize];

    id<MTLBuffer> scratch_buffer =
        [device newBufferWithLength:accel_sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> command_encoder =
        [command_buffer accelerationStructureCommandEncoder];

    // Readback buffer to get the compacted size
    id<MTLBuffer> compacted_size_buffer =
        [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // Queue the build
    [command_encoder buildAccelerationStructure:scratch_as
                                     descriptor:desc
                                  scratchBuffer:scratch_buffer
                            scratchBufferOffset:0];

    // Get the compacted size back from the build
    [command_encoder writeCompactedAccelerationStructureSize:scratch_as
                                                    toBuffer:compacted_size_buffer
                                                      offset:0];

    // Submit the buffer and wait for the build so we can read back the compact size
    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    uint32_t compact_size = *reinterpret_cast<uint32_t *>(compacted_size_buffer.contents);
    std::cout << "Compact size: " << compact_size << "b\n";

    // Now allocate the compact AS and compact the structure into it
    // For our single triangle AS this won't make it any smaller, but shows how it's done
    id<MTLAccelerationStructure> compact_as =
        [device newAccelerationStructureWithSize:compact_size];
    command_buffer = [command_queue commandBuffer];
    command_encoder = [command_buffer accelerationStructureCommandEncoder];

    [command_encoder copyAndCompactAccelerationStructure:scratch_as
                                 toAccelerationStructure:compact_as];
    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    return compact_as;
}
