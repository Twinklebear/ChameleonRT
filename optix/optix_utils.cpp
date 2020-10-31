#include "optix_utils.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <optix_stubs.h>
#include "util.h"

namespace optix {

Buffer::Buffer(size_t size) : buf_size(size)
{
    CHECK_CUDA(cudaMalloc(&ptr, buf_size));
}
Buffer::~Buffer()
{
    cudaFree(ptr);
}

Buffer::Buffer(Buffer &&b) : buf_size(b.buf_size), ptr(b.ptr)
{
    b.buf_size = 0;
    b.ptr = nullptr;
}
Buffer &Buffer::operator=(Buffer &&b)
{
    CHECK_CUDA(cudaFree(ptr));
    buf_size = b.buf_size;
    ptr = b.ptr;

    b.buf_size = 0;
    b.ptr = nullptr;

    return *this;
}

CUdeviceptr Buffer::device_ptr() const
{
    return reinterpret_cast<CUdeviceptr>(ptr);
}

size_t Buffer::size() const
{
    return buf_size;
}

void Buffer::upload(const void *data, size_t size)
{
    CHECK_CUDA(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
}

void Buffer::download(void *data, size_t size)
{
    CHECK_CUDA(cudaMemcpy(data, ptr, size, cudaMemcpyDeviceToHost));
}

void Buffer::clear()
{
    CHECK_CUDA(cudaMemset(ptr, 0, buf_size));
}

Texture2D::Texture2D(glm::uvec2 dims,
                     cudaChannelFormatDesc channel_format,
                     ColorSpace color_space)
    : tdims(dims), channel_format(channel_format)
{
    CHECK_CUDA(cudaMallocArray(&data, &channel_format, dims.x, dims.y));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = data;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.sRGB = color_space == SRGB ? 1 : 0;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 1;
    tex_desc.minMipmapLevelClamp = 1;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;

    CHECK_CUDA(cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));
}

Texture2D::~Texture2D()
{
    if (data) {
        cudaFreeArray(data);
        cudaDestroyTextureObject(texture);
    }
}

Texture2D::Texture2D(Texture2D &&t)
    : tdims(t.tdims), channel_format(t.channel_format), data(t.data), texture(t.texture)
{
    t.tdims = glm::uvec2(0);
    t.data = 0;
    t.texture = 0;
}

Texture2D &Texture2D::operator=(Texture2D &&t)
{
    if (data) {
        cudaFreeArray(data);
        cudaDestroyTextureObject(texture);
    }
    tdims = t.tdims;
    channel_format = t.channel_format;
    data = t.data;
    texture = t.texture;

    t.tdims = glm::uvec2(0);
    t.data = 0;
    t.texture = 0;
    return *this;
}

void Texture2D::upload(const uint8_t *buf)
{
    const size_t pixel_size =
        (channel_format.x + channel_format.y + channel_format.z + channel_format.w) / 8;
    const size_t pitch = pixel_size * tdims.x;
    CHECK_CUDA(
        cudaMemcpy2DToArray(data, 0, 0, buf, pitch, pitch, tdims.y, cudaMemcpyHostToDevice));
}

cudaTextureObject_t Texture2D::handle()
{
    return texture;
}

glm::uvec2 Texture2D::dims() const
{
    return tdims;
}

Geometry::Geometry(std::shared_ptr<Buffer> vertex_buf,
                   std::shared_ptr<Buffer> index_buf,
                   std::shared_ptr<Buffer> normal_buf,
                   std::shared_ptr<Buffer> uv_buf,
                   uint32_t geom_flags)
    : vertex_buf(vertex_buf),
      index_buf(index_buf),
      normal_buf(normal_buf),
      uv_buf(uv_buf),
      geom_flags(geom_flags),
      vertex_buf_ptr(vertex_buf->device_ptr())
{
}

OptixBuildInput Geometry::geom_desc() const
{
    OptixBuildInput desc = {};

    desc.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    desc.triangleArray.vertexBuffers = &vertex_buf_ptr;
    desc.triangleArray.numVertices = vertex_buf->size() / sizeof(glm::vec3);
    desc.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    desc.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);

    desc.triangleArray.indexBuffer = index_buf->device_ptr();
    desc.triangleArray.numIndexTriplets = index_buf->size() / sizeof(glm::uvec3);
    desc.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    desc.triangleArray.indexStrideInBytes = sizeof(glm::uvec3);

    desc.triangleArray.flags = &geom_flags;
    desc.triangleArray.numSbtRecords = 1;
    return desc;
}

TriangleMesh::TriangleMesh(std::vector<Geometry> &geoms, uint32_t build_flags)
    : build_flags(build_flags), geometries(geoms)
{
    build_inputs.reserve(geometries.size());
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(build_inputs),
                   [](const Geometry &g) { return g.geom_desc(); });
}

void TriangleMesh::enqueue_build(OptixDeviceContext &device, CUstream &stream)
{
    OptixAccelBuildOptions opts = {};
    opts.buildFlags = build_flags;
    opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    opts.motionOptions.numKeys = 1;

    OptixAccelBufferSizes buf_sizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        device, &opts, build_inputs.data(), build_inputs.size(), &buf_sizes));

#if 0
	std::cout << "BLAS will use output space of "
		<< pretty_print_count(buf_sizes.outputSizeInBytes)
		<< " plus scratch of " << pretty_print_count(buf_sizes.tempSizeInBytes) << "\n";
#endif

    build_output = Buffer(buf_sizes.outputSizeInBytes);
    scratch = Buffer(buf_sizes.tempSizeInBytes);

    // Now build the BLAS and query the info about the compacted size
    post_build_info = Buffer(sizeof(uint64_t));
    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = post_build_info.device_ptr();

    CHECK_OPTIX(optixAccelBuild(device,
                                stream,
                                &opts,
                                build_inputs.data(),
                                build_inputs.size(),
                                scratch.device_ptr(),
                                scratch.size(),
                                build_output.device_ptr(),
                                build_output.size(),
                                &as_handle,
                                &emit_desc,
                                1));
}

void TriangleMesh::enqueue_compaction(OptixDeviceContext &device, CUstream &stream)
{
    uint64_t compacted_size = 0;
    post_build_info.download(&compacted_size, sizeof(uint64_t));

#if 0
	std::cout << "BLAS will compact to " << pretty_print_count(compacted_size) << "\n";
#endif
    bvh = optix::Buffer(compacted_size);

    CHECK_OPTIX(optixAccelCompact(
        device, stream, as_handle, bvh.device_ptr(), bvh.size(), &as_handle));
}

void TriangleMesh::finalize()
{
    if (build_flags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
        build_output = Buffer();
    }
    scratch = Buffer();
    post_build_info = Buffer();
}

OptixTraversableHandle TriangleMesh::handle()
{
    return as_handle;
}

TopLevelBVH::TopLevelBVH(std::shared_ptr<Buffer> instance_buf,
                         const std::vector<Instance> &instances,
                         uint32_t build_flags)
    : build_flags(build_flags), instance_buf(instance_buf), instances(instances)
{
    geom_desc.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    geom_desc.instanceArray.instances = instance_buf->device_ptr();
    geom_desc.instanceArray.numInstances = instance_buf->size() / sizeof(OptixInstance);
}

void TopLevelBVH::enqueue_build(OptixDeviceContext &device, CUstream &stream)
{
    // TODO: Compared to DXR this is actually directly the same as the bottomlevel build,
    // so they can re-use the same code path.
    OptixAccelBuildOptions opts = {};
    opts.buildFlags = build_flags;
    opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    opts.motionOptions.numKeys = 1;

    OptixAccelBufferSizes buf_sizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(device, &opts, &geom_desc, 1, &buf_sizes));
#if 0
	std::cout << "BLAS will use output space of "
		<< pretty_print_count(buf_sizes.outputSizeInBytes)
		<< " plus scratch of " << pretty_print_count(buf_sizes.tempSizeInBytes) << "\n";
#endif

    build_output = Buffer(buf_sizes.outputSizeInBytes);
    scratch = Buffer(buf_sizes.tempSizeInBytes);

    // Now build the BLAS and query the info about the compacted size
    post_build_info = Buffer(sizeof(uint64_t));
    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = post_build_info.device_ptr();

    CHECK_OPTIX(optixAccelBuild(device,
                                stream,
                                &opts,
                                &geom_desc,
                                1,
                                scratch.device_ptr(),
                                scratch.size(),
                                build_output.device_ptr(),
                                build_output.size(),
                                &as_handle,
                                &emit_desc,
                                1));
}

void TopLevelBVH::enqueue_compaction(OptixDeviceContext &device, CUstream &stream)
{
    uint64_t compacted_size = 0;
    post_build_info.download(&compacted_size, sizeof(uint64_t));

#if 0
	std::cout << "TLAS will compact to " << pretty_print_count(compacted_size) << "\n";
#endif
    bvh = optix::Buffer(compacted_size);

    CHECK_OPTIX(optixAccelCompact(
        device, stream, as_handle, bvh.device_ptr(), bvh.size(), &as_handle));
}

void TopLevelBVH::finalize()
{
    if (build_flags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
        build_output = Buffer();
    }
    scratch = Buffer();
    post_build_info = Buffer();
}

size_t TopLevelBVH::num_instances() const
{
    return geom_desc.instanceArray.numInstances;
}

OptixTraversableHandle TopLevelBVH::handle()
{
    return as_handle;
}

Module::Module(OptixDeviceContext &device,
               const unsigned char *ptx,
               size_t ptex_len,
               const OptixModuleCompileOptions &compile_opts,
               const OptixPipelineCompileOptions &pipeline_opts)
{
    char log[2048] = {0};
    size_t log_size = sizeof(log);
    CHECK_OPTIX(optixModuleCreateFromPTX(device,
                                         &compile_opts,
                                         &pipeline_opts,
                                         reinterpret_cast<const char *>(ptx),
                                         ptex_len,
                                         log,
                                         &log_size,
                                         &module));
#if 0
	if (log_size > 0) {
		std::cout << log << "\n";
	}
#endif
}

Module::~Module()
{
    optixModuleDestroy(module);
}

OptixProgramGroup Module::create_program(OptixDeviceContext &device,
                                         OptixProgramGroupDesc &desc)
{
    OptixProgramGroupOptions opts = {};
    OptixProgramGroup prog;
    char log[2048];
    size_t log_size = sizeof(log);
    CHECK_OPTIX(optixProgramGroupCreate(device, &desc, 1, &opts, log, &log_size, &prog));
#if 0
	if (log_size > 0) {
		std::cout << log << "\n";
	}
#endif
    return prog;
}

OptixProgramGroup Module::create_raygen(OptixDeviceContext &device,
                                        const std::string &function)
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = function.c_str();
    return create_program(device, desc);
}

OptixProgramGroup Module::create_miss(OptixDeviceContext &device, const std::string &function)
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module;
    desc.miss.entryFunctionName = function.c_str();
    return create_program(device, desc);
}

OptixProgramGroup Module::create_hitgroup(OptixDeviceContext &device,
                                          const std::string &closest_hit,
                                          const std::string &any_hit,
                                          const std::string &intersection)
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleCH = module;
    desc.hitgroup.entryFunctionNameCH = closest_hit.c_str();

    if (!any_hit.empty()) {
        desc.hitgroup.moduleAH = module;
        desc.hitgroup.entryFunctionNameAH = any_hit.c_str();
    }

    if (!intersection.empty()) {
        desc.hitgroup.moduleIS = module;
        desc.hitgroup.entryFunctionNameIS = intersection.c_str();
    }

    return create_program(device, desc);
}

OptixPipeline compile_pipeline(OptixDeviceContext &device,
                               const OptixPipelineCompileOptions &compile_opts,
                               const OptixPipelineLinkOptions &link_opts,
                               const std::vector<OptixProgramGroup> &programs)
{
    OptixPipeline pipeline;
    char log[2048];
    size_t log_size = sizeof(log);
    CHECK_OPTIX(optixPipelineCreate(device,
                                    &compile_opts,
                                    &link_opts,
                                    programs.data(),
                                    programs.size(),
                                    log,
                                    &log_size,
                                    &pipeline));
#if 0
	if (log_size > 0) {
		std::cout << log << "\n";
	}
#endif
    return pipeline;
}

ShaderRecord::ShaderRecord(const std::string &name,
                           OptixProgramGroup program,
                           size_t param_size)
    : name(name), program(program), param_size(param_size)
{
}

ShaderTable::ShaderTable(const ShaderRecord &raygen_record,
                         const std::vector<ShaderRecord> &miss_records,
                         const std::vector<ShaderRecord> &hitgroup_records)
{
    const size_t raygen_entry_size = align_to(
        raygen_record.param_size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);

    size_t miss_entry_size = 0;
    for (const auto &m : miss_records) {
        miss_entry_size = std::max(
            miss_entry_size,
            align_to(m.param_size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT));
    }

    size_t hitgroup_entry_size = 0;
    for (const auto &h : hitgroup_records) {
        hitgroup_entry_size = std::max(
            hitgroup_entry_size,
            align_to(h.param_size + OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT));
    }

    const size_t sbt_size = raygen_entry_size + miss_records.size() * miss_entry_size +
                            hitgroup_records.size() * hitgroup_entry_size;

    shader_table = Buffer(sbt_size);
    cpu_shader_table.resize(sbt_size, 0);

    binding_table.raygenRecord = shader_table.device_ptr();

    binding_table.missRecordBase = binding_table.raygenRecord + raygen_entry_size;
    binding_table.missRecordStrideInBytes = miss_entry_size;
    binding_table.missRecordCount = miss_records.size();

    binding_table.hitgroupRecordBase =
        binding_table.missRecordBase + miss_records.size() * miss_entry_size;
    binding_table.hitgroupRecordStrideInBytes = hitgroup_entry_size;
    binding_table.hitgroupRecordCount = hitgroup_records.size();

    size_t offset = 0;
    record_offsets[raygen_record.name] = offset;
    optixSbtRecordPackHeader(raygen_record.program, &cpu_shader_table[offset]);
    offset += raygen_entry_size;

    for (const auto &m : miss_records) {
        record_offsets[m.name] = offset;
        optixSbtRecordPackHeader(m.program, &cpu_shader_table[offset]);
        offset += miss_entry_size;
    }

    for (const auto &h : hitgroup_records) {
        record_offsets[h.name] = offset;
        optixSbtRecordPackHeader(h.program, &cpu_shader_table[offset]);
        offset += hitgroup_entry_size;
    }
}

uint8_t *ShaderTable::get_shader_record(const std::string &shader)
{
    return &cpu_shader_table[record_offsets[shader]];
}

void ShaderTable::upload()
{
    shader_table.upload(cpu_shader_table);
}

const OptixShaderBindingTable &ShaderTable::table()
{
    return binding_table;
}

ShaderTableBuilder &ShaderTableBuilder::set_raygen(const std::string &name,
                                                   OptixProgramGroup program,
                                                   size_t param_size)
{
    raygen_record = ShaderRecord(name, program, param_size);
    return *this;
}

ShaderTableBuilder &ShaderTableBuilder::add_miss(const std::string &name,
                                                 OptixProgramGroup program,
                                                 size_t param_size)
{
    miss_records.emplace_back(name, program, param_size);
    return *this;
}

ShaderTableBuilder &ShaderTableBuilder::add_hitgroup(const std::string &name,
                                                     OptixProgramGroup program,
                                                     size_t param_size)
{
    hitgroup_records.emplace_back(name, program, param_size);
    return *this;
}

ShaderTable ShaderTableBuilder::build()
{
    return ShaderTable(raygen_record, miss_records, hitgroup_records);
}
}
