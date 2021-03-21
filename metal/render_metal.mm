#include "render_metal.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <Metal/Metal.h>
#include "metalrt_utils.h"
#include "render_metal_embedded_metallib.h"
#include "shader_types.h"
#include "util.h"
#include <glm/ext.hpp>

RenderMetal::RenderMetal(std::shared_ptr<metal::Context> ctx) : context(ctx)
{
    @autoreleasepool {
        shader_library = std::make_shared<metal::ShaderLibrary>(
            *context, render_metal_metallib, sizeof(render_metal_metallib));

        pipeline = std::make_shared<metal::ComputePipeline>(
            *context, shader_library->new_function(@"raygen"));

        native_display = true;
    }
}

RenderMetal::RenderMetal() : RenderMetal(std::make_shared<metal::Context>())
{
    native_display = false;
}

std::string RenderMetal::name()
{
    return "Metal Ray Tracing";
}

void RenderMetal::initialize(const int fb_width, const int fb_height)
{
    @autoreleasepool {
        frame_id = 0;
        img.resize(fb_width * fb_height);

        render_target = std::make_shared<metal::Texture2D>(*context,
                                                           fb_width,
                                                           fb_height,
                                                           MTLPixelFormatRGBA8Unorm,
                                                           MTLTextureUsageShaderWrite);

        accum_buffer = std::make_shared<metal::Texture2D>(*context,
                                                          fb_width,
                                                          fb_height,
                                                          MTLPixelFormatRGBA32Float,
                                                          MTLTextureUsageShaderWrite);

#ifdef REPORT_RAY_STATS
        ray_stats_readback.resize(fb_width * fb_height);
        ray_stats = std::make_shared<metal::Texture2D>(
            *context, fb_width, fb_height, MTLPixelFormatR16Uint, MTLTextureUsageShaderWrite);
#endif
    }
}

void RenderMetal::set_scene(const Scene &scene)
{
    @autoreleasepool {
        // Create a heap to hold all the data we'll need to upload
        allocate_heap(scene);

        // Upload the geometry for each mesh and build its BLAS
        std::vector<std::shared_ptr<metal::BottomLevelBVH>> meshes = build_meshes(scene);

        bvh = std::make_shared<metal::TopLevelBVH>(
            scene.parameterized_meshes, scene.instances, meshes);
        {
            id<MTLCommandBuffer> command_buffer = context->command_buffer();
            id<MTLAccelerationStructureCommandEncoder> command_encoder =
                [command_buffer accelerationStructureCommandEncoder];

            bvh->enqueue_build(*context, command_encoder);

            [command_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            command_buffer = context->command_buffer();
            command_encoder = [command_buffer accelerationStructureCommandEncoder];

            bvh->enqueue_compaction(*context, command_encoder);

            [command_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }

        // Upload the parameterized mesh material id buffers to the heap
        for (const auto &pm : scene.parameterized_meshes) {
            metal::Buffer upload(*context,
                                 sizeof(uint32_t) * pm.material_ids.size(),
                                 MTLResourceStorageModeManaged);
            std::memcpy(upload.data(), pm.material_ids.data(), upload.size());
            upload.mark_modified();

            auto material_id_buffer = std::make_shared<metal::Buffer>(
                *data_heap, upload.size(), MTLResourceStorageModePrivate);
            @autoreleasepool {
                id<MTLCommandBuffer> command_buffer = context->command_buffer();
                id<MTLBlitCommandEncoder> blit_encoder = command_buffer.blitCommandEncoder;

                [blit_encoder copyFromBuffer:upload.buffer
                                sourceOffset:0
                                    toBuffer:material_id_buffer->buffer
                           destinationOffset:0
                                        size:material_id_buffer->size()];

                [blit_encoder endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];

                parameterized_mesh_material_ids.push_back(material_id_buffer);
            }
        }

        // Build the argument buffer for the instance. Each instance is passed its
        // inverse object transform (not provided by Metal), the list of geometry IDs
        // that make up its mesh, and a list of material IDs for each geometry
        {
            metal::ArgumentEncoderBuilder args_builder(*context);
            args_builder.add_constant(0, MTLDataTypeFloat4x4)
                .add_buffer(1, MTLArgumentAccessReadOnly)
                .add_buffer(2, MTLArgumentAccessReadOnly);

            const size_t instance_args_size = args_builder.encoded_length();

            instance_args_buffer =
                std::make_shared<metal::Buffer>(*context,
                                                scene.instances.size() * instance_args_size,
                                                MTLResourceStorageModeManaged);

            size_t offset = 0;
            for (size_t i = 0; i < scene.instances.size(); ++i) {
                const auto &inst = scene.instances[i];
                auto encoder = args_builder.encoder_for_buffer(*instance_args_buffer, offset);
                glm::mat4 *inverse_tfm =
                    reinterpret_cast<glm::mat4 *>(encoder->constant_data_at(0));
                *inverse_tfm = glm::inverse(scene.instances[i].transform);

                const auto &pm = scene.parameterized_meshes[inst.parameterized_mesh_id];
                const auto &mesh = bvh->meshes[pm.mesh_id];

                encoder->set_buffer(*mesh->geometry_id_buffer, 0, 1);
                encoder->set_buffer(
                    *parameterized_mesh_material_ids[inst.parameterized_mesh_id], 0, 2);

                offset += instance_args_size;
            }
            instance_args_buffer->mark_modified();
        }

        // Upload the material data
        material_buffer =
            std::make_shared<metal::Buffer>(*context,
                                            sizeof(DisneyMaterial) * scene.materials.size(),
                                            MTLResourceStorageModeManaged);
        std::memcpy(material_buffer->data(), scene.materials.data(), material_buffer->size());
        material_buffer->mark_modified();

        // Upload the lights data
        light_buffer = std::make_shared<metal::Buffer>(
            *context, sizeof(QuadLight) * scene.lights.size(), MTLResourceStorageModeManaged);
        std::memcpy(light_buffer->data(), scene.lights.data(), light_buffer->size());
        light_buffer->mark_modified();

        upload_textures(scene.textures);

        // Pass the handles of the textures through an argument buffer
        {
            metal::ArgumentEncoderBuilder args_builder(*context);
            args_builder.add_texture(0, MTLArgumentAccessReadOnly);
            const size_t tex_args_size = args_builder.encoded_length();

            texture_arg_buffer = std::make_shared<metal::Buffer>(
                *context, tex_args_size * textures.size(), MTLResourceStorageModeManaged);

            size_t tex_args_offset = 0;
            for (const auto &t : textures) {
                auto encoder =
                    args_builder.encoder_for_buffer(*texture_arg_buffer, tex_args_offset);
                encoder->set_texture(*t, 0);
                tex_args_offset += tex_args_size;
            }
        }
        texture_arg_buffer->mark_modified();
    }
}

RenderStats RenderMetal::render(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy,
                                const bool camera_changed,
                                const bool readback_framebuffer)
{
    @autoreleasepool {
        using namespace std::chrono;
        RenderStats stats;

        if (camera_changed) {
            frame_id = 0;
        }

        ViewParams view_params = compute_view_parameters(pos, dir, up, fovy);

        auto start = high_resolution_clock::now();
        id<MTLCommandBuffer> command_buffer = context->command_buffer();
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setTexture:render_target->texture atIndex:0];
        [command_encoder setTexture:accum_buffer->texture atIndex:1];
#ifdef REPORT_RAY_STATS
        [command_encoder setTexture:ray_stats->texture atIndex:2];
#endif

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
        [command_encoder setBuffer:bvh->instance_buffer->buffer offset:0 atIndex:3];
        [command_encoder setBuffer:instance_args_buffer->buffer offset:0 atIndex:4];
        [command_encoder setBuffer:material_buffer->buffer offset:0 atIndex:5];
        [command_encoder setBuffer:texture_arg_buffer->buffer offset:0 atIndex:6];
        [command_encoder setBuffer:light_buffer->buffer offset:0 atIndex:7];
        [command_encoder useHeap:data_heap->heap];

        [command_encoder setComputePipelineState:pipeline->pipeline];

        // Use Metal's non-uniform dispatch support to divide up into 16x16 thread groups
        const glm::uvec2 fb_dims = render_target->dims();
        [command_encoder dispatchThreads:MTLSizeMake(fb_dims.x, fb_dims.y, 1)
                   threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        auto end = high_resolution_clock::now();
        stats.render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-6;

        if (readback_framebuffer || !native_display) {
            render_target->readback(img.data());
        }
#if REPORT_RAY_STATS
        ray_stats->readback(ray_stats_readback.data());

        const uint64_t total_rays = std::accumulate(
            ray_stats_readback.begin(),
            ray_stats_readback.end(),
            uint64_t(0),
            [](const uint64_t &total, const uint16_t &c) { return total + c; });
        stats.rays_per_second = total_rays / (stats.render_time * 1.0e-3);
#endif

        ++frame_id;
        return stats;
    }
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
    view_params.num_lights = light_buffer->size() / sizeof(QuadLight);

    return view_params;
}

void RenderMetal::allocate_heap(const Scene &scene)
{
    @autoreleasepool {
        metal::HeapBuilder heap_builder(*context);
        // Allocate enough room to store the data for each mesh
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
                if (!g.normals.empty()) {
                    heap_builder.add_buffer(sizeof(glm::vec3) * g.normals.size(),
                                            MTLResourceStorageModePrivate);
                }
                if (!g.uvs.empty()) {
                    heap_builder.add_buffer(sizeof(glm::vec2) * g.uvs.size(),
                                            MTLResourceStorageModePrivate);
                }
            }
        }

        // Allocate room for the parameterized mesh's material ID lists
        for (const auto &pm : scene.parameterized_meshes) {
            heap_builder.add_buffer(sizeof(uint32_t) * pm.material_ids.size(),
                                    MTLResourceStorageModePrivate);
        }

        // Reserve space for the texture data in the heap
        for (const auto &t : scene.textures) {
            MTLPixelFormat format = t.color_space == LINEAR ? MTLPixelFormatRGBA8Unorm
                                                            : MTLPixelFormatRGBA8Unorm_sRGB;
            heap_builder.add_texture2d(t.width, t.height, format, MTLTextureUsageShaderRead);
        }

        data_heap = heap_builder.build();
    }
}

std::vector<std::shared_ptr<metal::BottomLevelBVH>> RenderMetal::build_meshes(
    const Scene &scene)
{
    @autoreleasepool {
        // We also need to build a list of global geometry indices for each mesh, since
        // all the geometry info will be flattened into a single buffer
        uint32_t total_geometries = 0;
        std::vector<std::shared_ptr<metal::BottomLevelBVH>> meshes;

        for (const auto &m : scene.meshes) {
            // Upload the mesh geometry ids first
            std::shared_ptr<metal::Buffer> geom_id_buffer;
            {
                metal::Buffer geom_id_upload(*context,
                                             sizeof(uint32_t) * m.geometries.size(),
                                             MTLResourceStorageModeManaged);
                uint32_t *geom_ids = reinterpret_cast<uint32_t *>(geom_id_upload.data());
                for (uint32_t i = 0; i < m.geometries.size(); ++i) {
                    geom_ids[i] = total_geometries++;
                }
                geom_id_upload.mark_modified();

                geom_id_buffer = std::make_shared<metal::Buffer>(
                    *data_heap, geom_id_upload.size(), MTLResourceStorageModePrivate);

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

                // Allocate the buffers from the heap and copy the data into them
                auto vertex_buffer = std::make_shared<metal::Buffer>(
                    *data_heap, vertex_upload.size(), MTLResourceStorageModePrivate);

                auto index_buffer = std::make_shared<metal::Buffer>(
                    *data_heap, index_upload.size(), MTLResourceStorageModePrivate);

                std::shared_ptr<metal::Buffer> normal_upload = nullptr;
                std::shared_ptr<metal::Buffer> normal_buffer = nullptr;
                if (!g.normals.empty()) {
                    normal_upload =
                        std::make_shared<metal::Buffer>(*context,
                                                        sizeof(glm::vec3) * g.normals.size(),
                                                        MTLResourceStorageModeManaged);
                    std::memcpy(
                        normal_upload->data(), g.normals.data(), normal_upload->size());
                    normal_upload->mark_modified();

                    normal_buffer = std::make_shared<metal::Buffer>(
                        *data_heap, normal_upload->size(), MTLResourceStorageModePrivate);
                }

                std::shared_ptr<metal::Buffer> uv_upload = nullptr;
                std::shared_ptr<metal::Buffer> uv_buffer = nullptr;
                if (!g.uvs.empty()) {
                    uv_upload =
                        std::make_shared<metal::Buffer>(*context,
                                                        sizeof(glm::vec2) * g.uvs.size(),
                                                        MTLResourceStorageModeManaged);
                    std::memcpy(uv_upload->data(), g.uvs.data(), uv_upload->size());
                    uv_upload->mark_modified();

                    uv_buffer = std::make_shared<metal::Buffer>(
                        *data_heap, uv_upload->size(), MTLResourceStorageModePrivate);
                }

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

                if (normal_upload) {
                    [blit_encoder copyFromBuffer:normal_upload->buffer
                                    sourceOffset:0
                                        toBuffer:normal_buffer->buffer
                               destinationOffset:0
                                            size:normal_buffer->size()];
                }

                if (uv_upload) {
                    [blit_encoder copyFromBuffer:uv_upload->buffer
                                    sourceOffset:0
                                        toBuffer:uv_buffer->buffer
                               destinationOffset:0
                                            size:uv_buffer->size()];
                }

                [blit_encoder endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];

                geometries.emplace_back(vertex_buffer, index_buffer, normal_buffer, uv_buffer);
            }

            // Build the BLAS
            auto mesh = std::make_shared<metal::BottomLevelBVH>(geometries, geom_id_buffer);
            id<MTLCommandBuffer> command_buffer = context->command_buffer();
            id<MTLAccelerationStructureCommandEncoder> command_encoder =
                [command_buffer accelerationStructureCommandEncoder];

            mesh->enqueue_build(*context, command_encoder);

            [command_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            command_buffer = context->command_buffer();
            command_encoder = [command_buffer accelerationStructureCommandEncoder];

            mesh->enqueue_compaction(*context, command_encoder);

            [command_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            meshes.push_back(mesh);
        }

        // Build the argument buffer for each geometry
        metal::ArgumentEncoderBuilder geom_args_encoder_builder(*context);
        geom_args_encoder_builder.add_buffer(0, MTLArgumentAccessReadOnly)
            .add_buffer(1, MTLArgumentAccessReadOnly)
            .add_buffer(2, MTLArgumentAccessReadOnly)
            .add_buffer(3, MTLArgumentAccessReadOnly)
            .add_constant(4, MTLDataTypeUInt)
            .add_constant(5, MTLDataTypeUInt);

        const uint32_t geom_args_size = geom_args_encoder_builder.encoded_length();
        geometry_args_buffer = std::make_shared<metal::Buffer>(
            *context, geom_args_size * total_geometries, MTLResourceStorageModeManaged);

        // Write the geometry arguments to the buffer
        size_t geom_args_offset = 0;
        for (const auto &m : meshes) {
            // Write the geometry data arguments
            for (const auto &g : m->geometries) {
                auto encoder = geom_args_encoder_builder.encoder_for_buffer(
                    *geometry_args_buffer, geom_args_offset);
                encoder->set_buffer(*g.vertex_buf, 0, 0);
                encoder->set_buffer(*g.index_buf, 0, 1);

                uint32_t *num_normals =
                    reinterpret_cast<uint32_t *>(encoder->constant_data_at(4));
                if (g.normal_buf) {
                    encoder->set_buffer(*g.normal_buf, 0, 2);
                    *num_normals = g.normal_buf->size() / sizeof(glm::vec3);
                } else {
                    *num_normals = 0;
                }

                uint32_t *num_uvs = reinterpret_cast<uint32_t *>(encoder->constant_data_at(5));
                if (g.uv_buf) {
                    encoder->set_buffer(*g.uv_buf, 0, 3);
                    *num_uvs = g.uv_buf->size() / sizeof(glm::vec2);
                } else {
                    *num_uvs = 0;
                }

                geom_args_offset += geom_args_size;
            }
        }
        geometry_args_buffer->mark_modified();

        return meshes;
    }
}

void RenderMetal::upload_textures(const std::vector<Image> &scene_textures)
{
    @autoreleasepool {
        for (const auto &t : scene_textures) {
            const MTLPixelFormat format = t.color_space == LINEAR
                                              ? MTLPixelFormatRGBA8Unorm
                                              : MTLPixelFormatRGBA8Unorm_sRGB;

            metal::Texture2D upload(
                *context, t.width, t.height, format, MTLTextureUsageShaderRead);
            upload.upload(t.img.data());

            // Allocate a texture from the heap and copy into it
            auto heap_tex = std::make_shared<metal::Texture2D>(
                *data_heap, t.width, t.height, format, MTLTextureUsageShaderRead);

            id<MTLCommandBuffer> command_buffer = context->command_buffer();
            id<MTLBlitCommandEncoder> blit_encoder = command_buffer.blitCommandEncoder;

            [blit_encoder copyFromTexture:upload.texture toTexture:heap_tex->texture];

            [blit_encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            textures.push_back(heap_tex);
        }
    }
}

