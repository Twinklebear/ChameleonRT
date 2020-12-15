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
    // Create a heap to hold all the geometry buffers and mesh geometry ID buffers
    std::vector<std::vector<uint32_t>> mesh_geometry_ids;
    {
        metal::HeapBuilder heap_builder(*context);
        for (const auto &m : scene.meshes) {
            // Also get enough space to store the mesh's gometry indices buffer
            heap_builder.add_buffer(sizeof(uint32_t) * m.geometries.size(),
                                    MTLResourceStorageModePrivate);

            for (const auto &g : m.geometries) {
                heap_builder
                    .add_buffer(sizeof(glm::vec3) * g.vertices.size(),
                                MTLResourceStorageModePrivate)
                    .add_buffer(sizeof(glm::uvec3) * g.indices.size(),
                                MTLResourceStorageModePrivate);
                /*
                if (!g.normals.empty()) {
                    heap_builder.add_buffer(sizeof(glm::vec3) * g.normals.size(),
                                            MTLResourceStorageModePrivate);
                }
                if (!g.uvs.empty()) {
                    heap_builder.add_buffer(sizeof(glm::vec2) * g.uvs.size(),
                                            MTLResourceStorageModePrivate);
                }
                */
            }
        }
        geometry_heap = heap_builder.build();
    }

    // Upload the geometry for each mesh and build its BLAS
    std::vector<std::shared_ptr<metal::BottomLevelBVH>> meshes;

    // We also need to build a list of global geometry indices for each mesh, since
    // all the geometry info will be flattened into a single buffer
    uint32_t total_geometries = 0;
    std::vector<std::shared_ptr<metal::Buffer>> mesh_geometry_id_buffers;

    for (const auto &m : scene.meshes) {
        // Upload the mesh geometry ids first
        {
            metal::Buffer geom_id_upload(*context,
                                         sizeof(uint32_t) * m.geometries.size(),
                                         MTLResourceStorageModeManaged);
            uint32_t *geom_ids = reinterpret_cast<uint32_t *>(geom_id_upload.data());
            for (uint32_t i = 0; i < m.geometries.size(); ++i) {
                geom_ids[i] = total_geometries + i;
            }
            geom_id_upload.mark_modified();

            auto geom_id_buffer = std::make_shared<metal::Buffer>(
                *geometry_heap, geom_id_upload.size(), MTLResourceStorageModePrivate);

            id<MTLCommandBuffer> command_buffer = context->command_buffer();
            id<MTLBlitCommandEncoder> blit_encoder = command_buffer.blitCommandEncoder;

            [blit_encoder copyFromBuffer:geom_id_upload.buffer
                            sourceOffset:0
                                toBuffer:geom_id_buffer->buffer
                       destinationOffset:0
                                    size:geom_id_buffer->size()];

            [blit_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            [command_buffer release];

            mesh_geometry_id_buffers.push_back(geom_id_buffer);
        }

        std::vector<metal::Geometry> geometries;
        for (const auto &g : m.geometries) {
            metal::Buffer vertex_upload(*context,
                                        sizeof(glm::vec3) * g.vertices.size(),
                                        MTLResourceStorageModeManaged);

            std::memcpy(vertex_upload.data(), g.vertices.data(), vertex_upload.size());
            vertex_upload.mark_modified();

            metal::Buffer index_upload(*context,
                                       sizeof(glm::uvec3) * g.indices.size(),
                                       MTLResourceStorageModeManaged);
            std::memcpy(index_upload.data(), g.indices.data(), index_upload.size());
            index_upload.mark_modified();

            // TODO: normals and uvs as well

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
        total_geometries += geometries.size();

        // Build the BLAS
        auto mesh = std::make_shared<metal::BottomLevelBVH>(geometries);
        id<MTLCommandBuffer> command_buffer = context->command_buffer();
        id<MTLAccelerationStructureCommandEncoder> command_encoder =
            [command_buffer accelerationStructureCommandEncoder];

        mesh->enqueue_build(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];

        command_buffer = context->command_buffer();
        command_encoder = [command_buffer accelerationStructureCommandEncoder];

        mesh->enqueue_compaction(*context, command_encoder);

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        [command_encoder release];
        [command_buffer release];

        meshes.push_back(mesh);
    }

    // Build the argument buffer for the mesh geometry IDs
    metal::ArgumentEncoderBuilder mesh_args_encoder_builder(*context);
    mesh_args_encoder_builder.add_buffer(0, MTLArgumentAccessReadOnly);

    const uint32_t mesh_args_size = mesh_args_encoder_builder.encoded_length();
    std::cout << "Mesh args size: " << mesh_args_size
              << "b, total arg buffer size: " << mesh_args_size * meshes.size() << "b\n";
    mesh_args_buffer = std::make_shared<metal::Buffer>(
        *context, mesh_args_size * meshes.size(), MTLResourceStorageModeManaged);

    // Build the argument buffer for each geometry
    metal::ArgumentEncoderBuilder geom_args_encoder_builder(*context);
    geom_args_encoder_builder.add_buffer(0, MTLArgumentAccessReadOnly)
        .add_buffer(1, MTLArgumentAccessReadOnly);
    // TODO: also normals, uvs

    const uint32_t geom_args_size = geom_args_encoder_builder.encoded_length();
    std::cout << "Geom args size: " << geom_args_size
              << "b, total arg buffer size: " << geom_args_size * total_geometries << "b\n";
    geometry_args_buffer = std::make_shared<metal::Buffer>(
        *context, geom_args_size * total_geometries, MTLResourceStorageModeManaged);

    // Write the geometry arguments to the buffer
    size_t mesh_args_offset = 0;
    size_t geom_args_offset = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        // Write the mesh geometry ID buffer
        {
            auto argument_encoder = mesh_args_encoder_builder.encoder_for_buffer(
                *mesh_args_buffer, mesh_args_offset);
            argument_encoder->set_buffer(*mesh_geometry_id_buffers[i], 0, 0);
            mesh_args_offset += mesh_args_size;
        }

        // Write the geometry data arguments
        const auto &m = meshes[i];
        for (const auto &g : m->geometries) {
            auto argument_encoder = geom_args_encoder_builder.encoder_for_buffer(
                *geometry_args_buffer, geom_args_offset);
            argument_encoder->set_buffer(*g.vertex_buf, 0, 0);
            argument_encoder->set_buffer(*g.index_buf, 0, 1);

            geom_args_offset += geom_args_size;
        }
    }
    mesh_args_buffer->mark_modified();
    geometry_args_buffer->mark_modified();

    std::vector<Instance> instances = scene.instances;
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

    // Compute and upload the inverse instance transform matrices since Metal doesn't provide
    // these. TODO: Later this will merge with the material ID list info for each instance
    instance_inverse_transforms_buffer = std::make_shared<metal::Buffer>(
        *context, instances.size() * sizeof(glm::mat4), MTLResourceStorageModeManaged);
    glm::mat4 *instance_inverse_transforms =
        reinterpret_cast<glm::mat4 *>(instance_inverse_transforms_buffer->data());
    for (size_t i = 0; i < instances.size(); ++i) {
        instance_inverse_transforms[i] = glm::inverse(instances[i].transform);
    }
    instance_inverse_transforms_buffer->mark_modified();
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
    for (auto &mesh : bvh->meshes) {
        [command_encoder useResource:mesh->bvh usage:MTLResourceUsageRead];
    }

    [command_encoder setBuffer:geometry_args_buffer->buffer offset:0 atIndex:2];
    [command_encoder setBuffer:mesh_args_buffer->buffer offset:0 atIndex:3];
    [command_encoder useHeap:geometry_heap->heap];

    [command_encoder setBuffer:bvh->instance_buffer->buffer offset:0 atIndex:4];
    [command_encoder setBuffer:instance_inverse_transforms_buffer->buffer offset:0 atIndex:5];

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

