#include "render_metal.h"
#include <iostream>
#include <stdexcept>
#include <Metal/Metal.h>
#include "metalrt_utils.h"
#include "render_metal_embedded_metallib.h"
#include "shader_types.h"
#include "util.h"
#include <glm/ext.hpp>

RenderMetal::RenderMetal()
{
    context = std::make_shared<metal::Context>();

    std::cout << "Selected Metal device " << context->device_name() << "\n";

    shader_library = std::make_shared<metal::ShaderLibrary>(
        *context, render_metal_metallib, sizeof(render_metal_metallib));

    // Setup the compute pipeline
    pipeline = std::make_shared<metal::ComputePipeline>(
        *context, shader_library->new_function(@"raygen"));
}

std::string RenderMetal::name()
{
    return "Metal Ray Tracing";
}

void RenderMetal::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    img.resize(fb_width * fb_height);

    render_target = std::make_shared<metal::Texture2D>(
        *context, fb_width, fb_height, MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderWrite);
}

void RenderMetal::set_scene(const Scene &scene)
{
    // TODO Testing: Just take the first mesh of the first instance for now
    const Geometry &geom = scene.meshes[scene.instances[0].mesh_id].geometries[0];

    // Create a heap to hold all the geometry buffers
    geometry_heap = metal::HeapBuilder(*context)
                        .add_buffer(sizeof(glm::vec3) * geom.vertices.size(),
                                    MTLResourceStorageModePrivate)
                        .add_buffer(sizeof(glm::uvec3) * geom.indices.size(),
                                    MTLResourceStorageModePrivate)
                        .build();

    std::vector<metal::Geometry> geometries;
    // Upload the data to staging and copy it into the heap
    {
        metal::Buffer vertex_upload(
            *context, sizeof(glm::vec3) * geom.vertices.size(), MTLResourceStorageModeManaged);
        std::cout << "vertex_upload size: " << vertex_upload.size() << "\n";

        std::memcpy(vertex_upload.data(), geom.vertices.data(), vertex_upload.size());
        vertex_upload.mark_modified();

        metal::Buffer index_upload(
            *context, sizeof(glm::uvec3) * geom.indices.size(), MTLResourceStorageModeManaged);
        std::memcpy(index_upload.data(), geom.indices.data(), index_upload.size());
        index_upload.mark_modified();

        // Allocate the buffers from the heap and copy the data into them
        auto vertex_buffer = std::make_shared<metal::Buffer>(
            *geometry_heap, vertex_upload.size(), MTLResourceStorageModePrivate);

        auto index_buffer = std::make_shared<metal::Buffer>(
            *geometry_heap, index_upload.size(), MTLResourceStorageModePrivate);

        id<MTLCommandBuffer> command_buffer = context->command_buffer();
        id<MTLBlitCommandEncoder> blit_encoder = command_buffer.blitCommandEncoder;

        [blit_encoder copyFromBuffer:vertex_upload.buffer
                        sourceOffset:0
                            toBuffer:vertex_buffer->buffer
                   destinationOffset:0
                                size:vertex_buffer->size()];

        [blit_encoder copyFromBuffer:index_upload.buffer
                        sourceOffset:0
                            toBuffer:index_buffer->buffer
                   destinationOffset:0
                                size:index_buffer->size()];

        [blit_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        [command_buffer release];

        geometries.emplace_back(vertex_buffer, index_buffer, nullptr, nullptr);
    }

    // Build argument buffer for the mesh
    metal::ArgumentEncoderBuilder argument_encoder_builder(*context);
    argument_encoder_builder.add_buffer(0, MTLArgumentAccessReadOnly)
        .add_buffer(1, MTLArgumentAccessReadOnly);

    // This will be the stride between consecutive geometries, though right now
    // we just have one so it's also the buffer size we need
    const uint32_t geom_args_stride = argument_encoder_builder.encoded_length();
    std::cout << "Geom args length: " << geom_args_stride << "b\n";
    geometry_args_buffer = std::make_shared<metal::Buffer>(
        *context, geom_args_stride, MTLResourceStorageModeManaged);

    // Write the arguments into the buffer
    auto argument_encoder =
        argument_encoder_builder.encoder_for_buffer(*geometry_args_buffer, 0);
    argument_encoder->set_buffer(*geometries[0].vertex_buf, 0, 0);
    argument_encoder->set_buffer(*geometries[0].index_buf, 0, 1);

    geometry_args_buffer->mark_modified();

    // Build the BLAS
    auto blas = std::make_shared<metal::BottomLevelBVH>(geometries);
    {
        id<MTLCommandBuffer> command_buffer = context->command_buffer();
        id<MTLAccelerationStructureCommandEncoder> command_encoder =
            [command_buffer accelerationStructureCommandEncoder];

        blas->enqueue_build(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];

        command_buffer = context->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];

        blas->enqueue_compaction(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];
    }

    // For now we're just drawing the first instance
    std::vector<Instance> instances = {scene.instances[0]};
    std::vector<std::shared_ptr<metal::BottomLevelBVH>> meshes = {blas};
    bvh = std::make_shared<metal::TopLevelBVH>(instances, meshes);

    {
        id<MTLCommandBuffer> command_buffer = context->command_buffer();
        id<MTLAccelerationStructureCommandEncoder> command_encoder =
            [command_buffer accelerationStructureCommandEncoder];

        bvh->enqueue_build(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];

        command_buffer = context->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];

        bvh->enqueue_compaction(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];
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

    id<MTLCommandBuffer> command_buffer = context->command_buffer();
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setTexture:render_target->texture atIndex:0];

    // Embed the view params in the command buffer
    [command_encoder setBytes:&view_params length:sizeof(ViewParams) atIndex:0];

    [command_encoder setAccelerationStructure:bvh->bvh atBufferIndex:1];
    // Also mark all BLAS's used
    // TODO: Seems like we can't do a similar heap thing for the BLAS's to mark
    // them all used at once?
    [command_encoder useResource:bvh->meshes[0]->bvh usage:MTLResourceUsageRead];

    [command_encoder setBuffer:geometry_args_buffer->buffer offset:0 atIndex:2];
    [command_encoder useHeap:geometry_heap->heap];

    [command_encoder setBuffer:bvh->instance_buffer->buffer offset:0 atIndex:3];

    [command_encoder setComputePipelineState:pipeline->pipeline];
    // TODO: Better thread group sizing here, this is a poor choice for utilization
    // but keeps the example simple
    const glm::uvec2 fb_dims = render_target->dims();
    [command_encoder dispatchThreadgroups:MTLSizeMake(fb_dims.x, fb_dims.y, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (readback_framebuffer || !native_display) {
        render_target->get_bytes(img.data());
    }

    [command_encoder release];
    [command_buffer release];

    ++frame_id;
    return stats;
}

ViewParams RenderMetal::compute_view_parameters(const glm::vec3 &pos,
                                                const glm::vec3 &dir,
                                                const glm::vec3 &up,
                                                const float fovy)
{
    const glm::uvec2 fb_dims = render_target->dims();
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

