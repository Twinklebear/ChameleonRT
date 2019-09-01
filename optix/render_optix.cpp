#include "render_optix.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include "optix_params.h"
#include "optix_utils.h"
#include "render_optix_embedded_ptx.h"
#include "types.h"
#include "util.h"

void log_callback(unsigned int level, const char *tag, const char *msg, void *)
{
    std::cout << "----\nOptiX Log Message (level " << level << "):\n"
              << "  Tag: " << tag << "\n"
              << "  Msg: " << msg << "\n----\n";
}

std::ostream &operator<<(std::ostream &os, const OptixStackSizes &s)
{
    os << "(cssRG: " << s.cssRG << ", "
       << "cssMS: " << s.cssMS << ", "
       << "cssCH: " << s.cssCH << ", "
       << "cssAH: " << s.cssAH << ", "
       << "cssIS: " << s.cssIS << ", "
       << "cssCC: " << s.cssCC << ", "
       << "dssDC: " << s.dssDC << ")";
    return os;
}

RenderOptiX::RenderOptiX()
{
    // Init CUDA and OptiX
    cudaFree(0);
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        throw std::runtime_error("No CUDA capable devices found!");
    }

    CHECK_OPTIX(optixInit());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&cuda_stream));

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    std::cout << "OptiX backend running on " << device_props.name << "\n";

    cuCtxGetCurrent(&cuda_context);

    CHECK_OPTIX(optixDeviceContextCreate(cuda_context, 0, &device));
    // TODO: set this val. based on the debug level
    CHECK_OPTIX(optixDeviceContextSetLogCallback(device, log_callback, nullptr, 0));

    launch_params = optix::Buffer(sizeof(LaunchParams));
}

RenderOptiX::~RenderOptiX()
{
    optixPipelineDestroy(pipeline);
    optixDeviceContextDestroy(device);
    cudaStreamDestroy(cuda_stream);
}

std::string RenderOptiX::name()
{
    return "OptiX";
}

void RenderOptiX::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    width = fb_width;
    height = fb_height;
    img.resize(fb_width * fb_height);

    framebuffer = optix::Buffer(img.size() * sizeof(uint32_t));
    accum_buffer = optix::Buffer(img.size() * sizeof(glm::vec4));
    accum_buffer.clear();
#ifdef REPORT_RAY_STATS
    ray_stats_buffer = optix::Buffer(img.size() * sizeof(uint16_t));
#endif
}

void RenderOptiX::set_scene(const Scene &scene)
{
    frame_id = 0;

    // TODO: We can actually run all these uploads and BVH builds in parallel
    // using cudaMemcpyAsync, and the builds in parallel on multiple streams.
    // Some helpers for managing the temp upload heap buf allocation and queuing of
    // the commands would help to make it easier to write the parallel load version which
    // won't exceed the GPU VRAM
    for (const auto &mesh : scene.meshes) {
        auto vertices = std::make_shared<optix::Buffer>(mesh.vertices.size() * sizeof(glm::vec3));
        vertices->upload(mesh.vertices);

        auto indices = std::make_shared<optix::Buffer>(mesh.indices.size() * sizeof(glm::uvec3));
        indices->upload(mesh.indices);

        std::shared_ptr<optix::Buffer> uvs = nullptr;
        if (!mesh.uvs.empty()) {
            uvs = std::make_shared<optix::Buffer>(mesh.uvs.size() * sizeof(glm::vec2));
            uvs->upload(mesh.uvs);
        }

        std::shared_ptr<optix::Buffer> normals = nullptr;
        if (!mesh.normals.empty()) {
            normals = std::make_shared<optix::Buffer>(mesh.normals.size() * sizeof(glm::vec3));
            normals->upload(mesh.normals);
        }

        // Build the bottom-level acceleration structure
        meshes.emplace_back(vertices,
                            indices,
                            normals,
                            uvs,
                            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

        meshes.back().enqueue_build(device, cuda_stream);
        sync_gpu();

        meshes.back().enqueue_compaction(device, cuda_stream);
        sync_gpu();

        meshes.back().finalize();
    }

    // Build the top-level acceleration structure over the instances
    // For now we don't really have "instances" since OBJ doesn't support this.
    // Note: both for DXR and OptiX we can just put all the triangle meshes
    // in one bottom-level AS, and use the geometry order indexed hit groups to
    // set the params properly for each geom. However, eventually I do plan to support
    // instancing so it's easiest to learn the whole path on a simple case.
    std::vector<OptixInstance> instances;
    instances.reserve(meshes.size());
    for (size_t i = 0; i < meshes.size(); ++i) {
        OptixInstance instance = {};

        // Same as DXR, the transform is 3x4 row-major
        instance.transform[0] = 1.f;
        instance.transform[4 + 1] = 1.f;
        instance.transform[2 * 4 + 2] = 1.f;

        instance.instanceId = scene.meshes[i].material_id;
        instance.sbtOffset = i * NUM_RAY_TYPES;
        instance.visibilityMask = 0xff;
        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instance.traversableHandle = meshes[i].handle();

        instances.push_back(instance);
    }

    // Upload the instance data to the GPU
    auto instance_buffer =
        std::make_shared<optix::Buffer>(instances.size() * sizeof(OptixInstance));
    instance_buffer->upload(instances);

    scene_bvh = optix::TopLevelBVH(instance_buffer, OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

    scene_bvh.enqueue_build(device, cuda_stream);
    sync_gpu();

    scene_bvh.enqueue_compaction(device, cuda_stream);
    sync_gpu();

    scene_bvh.finalize();

    const cudaChannelFormatDesc channel_format =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    for (const auto &t : scene.textures) {
        textures.emplace_back(glm::uvec2(t.width, t.height), channel_format);
        textures.back().upload(t.img.data());
    }

    std::vector<MaterialParams> material_params;
    material_params.reserve(scene.materials.size());
    for (const auto &m : scene.materials) {
        MaterialParams p;

        p.base_color = m.base_color;
        p.metallic = m.metallic;
        p.specular = m.specular;
        p.roughness = m.roughness;
        p.specular_tint = m.specular_tint;
        p.anisotropy = m.anisotropy;
        p.sheen = m.sheen;
        p.sheen_tint = m.sheen_tint;
        p.clearcoat = m.clearcoat;
        p.clearcoat_gloss = m.clearcoat_gloss;
        p.ior = m.ior;
        p.specular_transmission = m.specular_transmission;

        if (m.color_tex_id != -1) {
            p.has_color_tex = 1;
            p.color_texture = textures[m.color_tex_id].handle();
        } else {
            p.has_color_tex = 0;
        }

        material_params.push_back(p);
    }

    mat_params = optix::Buffer(material_params.size() * sizeof(MaterialParams));
    mat_params.upload(material_params);

    light_params = optix::Buffer(scene.lights.size() * sizeof(QuadLight));
    light_params.upload(scene.lights);

    build_raytracing_pipeline();
}

void RenderOptiX::build_raytracing_pipeline()
{
    // Setup the OptiX Module (DXR equivalent is the Shader Library)

    OptixPipelineCompileOptions pipeline_opts = {};
    pipeline_opts.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    // We pack a pointer to the payload stack var into 2 32bit ints
    pipeline_opts.numPayloadValues = 2;
    pipeline_opts.numAttributeValues = 2;
    pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_opts.pipelineLaunchParamsVariableName = "launch_params";

    optix::Module module(device,
                         render_optix_ptx,
                         sizeof(render_optix_ptx),
                         optix::DEFAULT_MODULE_COMPILE_OPTIONS,
                         pipeline_opts);

    // Now build the program pipeline

    // Make the raygen program
    OptixProgramGroup raygen_prog = module.create_raygen(device, "__raygen__perspective_camera");

    // Make the miss shader programs, one for each ray type
    std::array<OptixProgramGroup, 2> miss_progs = {
        module.create_miss(device, "__miss__miss"),
        module.create_miss(device, "__miss__occlusion_miss")};

    // Make the hit groups, for each ray type
    std::array<OptixProgramGroup, 2> hitgroup_progs = {
        module.create_hitgroup(device, "__closesthit__closest_hit"),
        module.create_hitgroup(device, "__closesthit__occlusion_hit")};

    // Combine the programs into a pipeline
    std::vector<OptixProgramGroup> pipeline_progs = {
        raygen_prog, miss_progs[0], miss_progs[1], hitgroup_progs[0], hitgroup_progs[1]};

    OptixPipelineLinkOptions link_opts = {};
    link_opts.maxTraceDepth = 1;
    link_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline = optix::compile_pipeline(device, pipeline_opts, link_opts, pipeline_progs);

    // We don't make any recursive any hit calls or use the callable shaders,
    // so we don't need extra stack space. TODO: Though 0 seems very small?
    CHECK_OPTIX(optixPipelineSetStackSize(pipeline, 0, 0, 0, 1));

    auto shader_table_builder =
        optix::ShaderTableBuilder()
            .set_raygen("perspective_camera", raygen_prog, sizeof(RayGenParams))
            .add_miss("miss", miss_progs[0], 0)
            .add_miss("occlusion_miss", miss_progs[1], 0);

    // Hitgroups for each instance
    for (size_t i = 0; i < meshes.size(); ++i) {
        shader_table_builder
            .add_hitgroup(
                "closest_hit_" + std::to_string(i), hitgroup_progs[0], sizeof(HitGroupParams))
            .add_hitgroup("occlusion_hit_" + std::to_string(i), hitgroup_progs[1], 0);
    }

    shader_table = shader_table_builder.build();

    {
        RayGenParams &params = shader_table.get_shader_params<RayGenParams>("perspective_camera");
        params.materials = mat_params.device_ptr();
        params.lights = light_params.device_ptr();
        params.num_lights = light_params.size() / sizeof(QuadLight);
    }

    for (size_t i = 0; i < meshes.size(); ++i) {
        HitGroupParams &params =
            shader_table.get_shader_params<HitGroupParams>("closest_hit_" + std::to_string(i));
        params.vertex_buffer = meshes[i].vertex_buf->device_ptr();
        params.index_buffer = meshes[i].index_buf->device_ptr();

        if (meshes[i].uv_buf) {
            params.uv_buffer = meshes[i].uv_buf->device_ptr();
        } else {
            params.uv_buffer = 0;
        }

        if (meshes[i].normal_buf) {
            params.normal_buffer = meshes[i].normal_buf->device_ptr();
        } else {
            params.normal_buffer = 0;
        }
    }

    shader_table.upload();

    // After compiling and linking the pipeline we don't need the module or programs
    optixProgramGroupDestroy(raygen_prog);
    for (size_t i = 0; i < miss_progs.size(); ++i) {
        optixProgramGroupDestroy(miss_progs[i]);
    }

    for (size_t i = 0; i < hitgroup_progs.size(); ++i) {
        optixProgramGroupDestroy(hitgroup_progs[i]);
    }
}

RenderStats RenderOptiX::render(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy,
                                const bool camera_changed)
{
    using namespace std::chrono;
    RenderStats stats;

    if (camera_changed) {
        frame_id = 0;
    }

    update_view_parameters(pos, dir, up, fovy);

    auto start = high_resolution_clock::now();

    CHECK_OPTIX(optixLaunch(pipeline,
                            cuda_stream,
                            launch_params.device_ptr(),
                            launch_params.size(),
                            &shader_table.table(),
                            width,
                            height,
                            1));

    // Sync with the GPU to ensure it actually finishes rendering
    sync_gpu();
    auto end = high_resolution_clock::now();
    stats.render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-6;

    framebuffer.download(img);
#ifdef REPORT_RAY_STATS
    std::vector<uint16_t> ray_counts(ray_stats_buffer.size() / sizeof(uint16_t), 0);
    ray_stats_buffer.download(ray_counts);
    const uint64_t total_rays =
        std::accumulate(ray_counts.begin(),
                        ray_counts.end(),
                        uint64_t(0),
                        [](const uint64_t &total, const uint16_t &c) { return total + c; });
    stats.rays_per_second = total_rays / (stats.render_time * 1.0e-3);
#endif

    ++frame_id;
    return stats;
}

void RenderOptiX::update_view_parameters(const glm::vec3 &pos,
                                         const glm::vec3 &dir,
                                         const glm::vec3 &up,
                                         const float fovy)
{
    LaunchParams params;

    glm::vec2 img_plane_size;
    img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
    img_plane_size.x = img_plane_size.y * static_cast<float>(width) / height;

    const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
    const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
    const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

    params.cam_pos = glm::vec4(pos, 0);
    params.cam_du = glm::vec4(dir_du, 0);
    params.cam_dv = glm::vec4(dir_dv, 0);
    params.cam_dir_top_left = glm::vec4(dir_top_left, 0);
    params.frame_id = frame_id;
    params.framebuffer = framebuffer.device_ptr();
    params.accum_buffer = accum_buffer.device_ptr();
#ifdef REPORT_RAY_STATS
    params.ray_stats_buffer = ray_stats_buffer.device_ptr();
#endif
    params.scene = scene_bvh.handle();

    launch_params.upload(&params, sizeof(LaunchParams));
}

void RenderOptiX::sync_gpu()
{
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
                  << std::endl
                  << std::flush;
        throw std::runtime_error("sync");
    }
}
