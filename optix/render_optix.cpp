#include <chrono>
#include <array>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "util.h"
#include "optix_utils.h"
#include "render_optix_embedded_ptx.h"
#include "render_optix.h"
#include "optix_params.h"

void log_callback(unsigned int level, const char *tag, const char *msg, void*) {
	std::cout << "----\nOptiX Log Message (level " << level << "):\n"
		<< "  Tag: " << tag << "\n"
		<< "  Msg: " << msg << "\n----\n";
}

std::ostream& operator<<(std::ostream &os, const OptixStackSizes &s) {
	os << "(cssRG: " << s.cssRG << ", "
		<< "cssMS: " << s.cssMS << ", "
		<< "cssCH: " << s.cssCH << ", "
		<< "cssAH: " << s.cssAH << ", "
		<< "cssIS: " << s.cssIS << ", "
		<< "cssCC: " << s.cssCC << ", "
		<< "dssDC: " << s.dssDC << ")";
	return os;
}

RenderOptiX::RenderOptiX() {
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

	CHECK_OPTIX(optixDeviceContextCreate(cuda_context, 0, &optix_context));
	// TODO: set this val. based on the debug level
	CHECK_OPTIX(optixDeviceContextSetLogCallback(optix_context, log_callback, nullptr, 4));

	// view params holds the camera params, frame id and pointers to the framebuffer and accum buffer
	cudaMalloc(&launch_params, sizeof(LaunchParams));
}

RenderOptiX::~RenderOptiX() {
	cudaFree(framebuffer);
	cudaFree(accum_buffer);
	cudaFree(launch_params);
	cudaFree(mat_params);

	cudaFree(vertices);
	cudaFree(indices);

	cudaFree(blas_buffer);

	cudaFree(instance_buffer);
	cudaFree(tlas_buffer);

	cudaFree(shader_table_data);

	// TODO: also release the accel structs and other stuff.
}

void RenderOptiX::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
	width = fb_width;
	height = fb_height;
	img.resize(fb_width * fb_height);

	if (framebuffer) {
		cudaFree(framebuffer);
		cudaFree(accum_buffer);
	}
	cudaMalloc(&framebuffer, img.size() * sizeof(uint32_t));
	cudaMalloc(&accum_buffer, img.size() * sizeof(glm::vec4));
}

void RenderOptiX::set_scene(const Scene &scene) {
	frame_id = 0;
	const auto &mesh = scene.meshes[0];

	cudaMalloc(&vertices, mesh.vertices.size() * sizeof(glm::vec3));
	cudaMemcpy(vertices, mesh.vertices.data(), mesh.vertices.size() * sizeof(glm::vec3),
			cudaMemcpyHostToDevice);

	cudaMalloc(&indices, mesh.indices.size() * sizeof(glm::uvec3));
	cudaMemcpy(indices, mesh.indices.data(), mesh.indices.size() * sizeof(glm::uvec3),
			cudaMemcpyHostToDevice);

	// Build the bottom-level acceleration structure
	{
		CUdeviceptr d_verts = reinterpret_cast<CUdeviceptr>(vertices);

		OptixBuildInput inputs = {};
		inputs.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		inputs.triangleArray.vertexBuffers = &d_verts;
		inputs.triangleArray.numVertices = mesh.vertices.size();
		inputs.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		inputs.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);

		inputs.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(indices);
		inputs.triangleArray.numIndexTriplets = mesh.indices.size();
		inputs.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		inputs.triangleArray.indexStrideInBytes = sizeof(glm::uvec3);

		uint32_t blas_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
		inputs.triangleArray.flags = &blas_flags;
		inputs.triangleArray.numSbtRecords = 1;

		OptixAccelBuildOptions opts = {};
		opts.buildFlags =  OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		opts.operation = OPTIX_BUILD_OPERATION_BUILD;
		opts.motionOptions.numKeys = 1;

		OptixAccelBufferSizes buf_sizes;
		CHECK_OPTIX(optixAccelComputeMemoryUsage(optix_context, &opts,
					&inputs, 1, &buf_sizes));

		std::cout << "BLAS will use output space of "
			<< pretty_print_count(buf_sizes.outputSizeInBytes)
			<< " plus scratch of " << pretty_print_count(buf_sizes.tempSizeInBytes) << "\n";

		void *build_output = nullptr;
		void *build_scratch = nullptr;
		cudaMalloc(&build_output, buf_sizes.outputSizeInBytes);
		cudaMalloc(&build_scratch, buf_sizes.tempSizeInBytes);

		// Now build the BLAS and query the info about the compacted size
		void *compacted_size_info = nullptr;
		cudaMalloc(&compacted_size_info, sizeof(uint64_t));
		OptixAccelEmitDesc post_info = {};
		post_info.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		post_info.result = reinterpret_cast<CUdeviceptr>(compacted_size_info);

		CHECK_OPTIX(optixAccelBuild(optix_context, cuda_stream, &opts, &inputs, 1,
					reinterpret_cast<CUdeviceptr>(build_scratch), buf_sizes.tempSizeInBytes,
					reinterpret_cast<CUdeviceptr>(build_output), buf_sizes.outputSizeInBytes,
					&blas_handle, &post_info, 1));

		// Wait for the build to complete before compacting
		sync_gpu();
		cudaFree(build_scratch);
		build_scratch = nullptr;

		uint64_t compacted_size = 0;
		cudaMemcpy(&compacted_size, compacted_size_info, sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaFree(compacted_size_info);

		std::cout << "BLAS will compact to " << pretty_print_count(compacted_size) << "\n";
		cudaMalloc(&blas_buffer, compacted_size);

		CHECK_OPTIX(optixAccelCompact(optix_context, cuda_stream, blas_handle,
					reinterpret_cast<CUdeviceptr>(blas_buffer), compacted_size,
					&blas_handle));
		sync_gpu();
		cudaFree(build_output);
	}

	// Build the top-level acceleration structure over the "instance"
	// Note: both for DXR and OptiX we can just put all the triangle meshes
	// in one bottom-level AS, and use the geometry order indexed hit groups to
	// set the params properly for each geom. However, eventually I do plan to support
	// instancing so it's easiest to learn the whole path on a simple case.
	{
		OptixInstance instance = {};

		// Same as DXR, the transform is 3x4 row-major
		instance.transform[0] = 1.f;
		instance.transform[4 + 1] = 1.f;
		instance.transform[2 * 4 + 2] = 1.f;

		instance.instanceId = 0;
		// sbt offset = DXR instanceContributionToHitGroupIndex
		instance.sbtOffset = 0;
		instance.visibilityMask = 0xff;
		instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
		instance.traversableHandle = blas_handle;

		// Upload the instance data to the GPU
		cudaMalloc(&instance_buffer, sizeof(OptixInstance));
		cudaMemcpy(instance_buffer, &instance, sizeof(OptixInstance), cudaMemcpyHostToDevice);

		OptixBuildInput inputs = {};
		inputs.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		inputs.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instance_buffer);
		inputs.instanceArray.numInstances = 1;

		OptixAccelBuildOptions opts = {};
		opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		opts.operation = OPTIX_BUILD_OPERATION_BUILD;
		opts.motionOptions.numKeys = 1;

		OptixAccelBufferSizes buf_sizes;
		CHECK_OPTIX(optixAccelComputeMemoryUsage(optix_context, &opts,
					&inputs, 1, &buf_sizes));

		std::cout << "TLAS will use output space of "
			<< pretty_print_count(buf_sizes.outputSizeInBytes)
			<< " plus scratch of " << pretty_print_count(buf_sizes.tempSizeInBytes) << "\n";

		void *build_output = nullptr;
		void *build_scratch = nullptr;
		cudaMalloc(&build_output, buf_sizes.outputSizeInBytes);
		cudaMalloc(&build_scratch, buf_sizes.tempSizeInBytes);

		// Now build the TLAS and query the info about the compacted size
		void *compacted_size_info = nullptr;
		cudaMalloc(&compacted_size_info, sizeof(uint64_t));
		OptixAccelEmitDesc post_info = {};
		post_info.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		post_info.result = reinterpret_cast<CUdeviceptr>(compacted_size_info);

		CHECK_OPTIX(optixAccelBuild(optix_context, cuda_stream, &opts, &inputs, 1,
					reinterpret_cast<CUdeviceptr>(build_scratch), buf_sizes.tempSizeInBytes,
					reinterpret_cast<CUdeviceptr>(build_output), buf_sizes.outputSizeInBytes,
					&tlas_handle, &post_info, 1));

		// Wait for the build to complete before compacting
		sync_gpu();
		cudaFree(build_scratch);
		build_scratch = nullptr;

		uint64_t compacted_size = 0;
		cudaMemcpy(&compacted_size, compacted_size_info, sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaFree(compacted_size_info);

		std::cout << "TLAS will compact to " << pretty_print_count(compacted_size) << "\n";
		cudaMalloc(&tlas_buffer, compacted_size);

		CHECK_OPTIX(optixAccelCompact(optix_context, cuda_stream, tlas_handle,
					reinterpret_cast<CUdeviceptr>(tlas_buffer), compacted_size,
					&tlas_handle));
		sync_gpu();
		cudaFree(build_output);
	}

	cudaMalloc(&mat_params, sizeof(DisneyMaterial));
	cudaMemcpy(mat_params, &scene.materials[mesh.material_id], sizeof(DisneyMaterial),
			cudaMemcpyHostToDevice);

	build_raytracing_pipeline();
}

void RenderOptiX::build_raytracing_pipeline() {
	// Setup the OptiX Module (DXR equivalent is the Shader Library)
	OptixModuleCompileOptions module_opts = {};
	module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT; 
	// TODO: pick these based on debug level in cmake
	module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	OptixPipelineCompileOptions pipeline_opts = {};
	pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	// We pack a pointer to the payload stack var into 2 32bit ints
	pipeline_opts.numPayloadValues = 2;
	pipeline_opts.numAttributeValues = 2;
	pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeline_opts.pipelineLaunchParamsVariableName = "launch_params";

	char log[2048] = {0};
	size_t log_size = sizeof(log);
	CHECK_OPTIX(optixModuleCreateFromPTX(optix_context, &module_opts, &pipeline_opts,
				reinterpret_cast<const char*>(render_optix_ptx), sizeof(render_optix_ptx),
				log, &log_size, &module));
	if (log_size > 0) {
		std::cout << log << "\n";
	}

	// Now build the program pipeline
	OptixProgramGroupOptions prog_opts = {};

	// Make the raygen program
	OptixProgramGroup raygen_prog;
	{
		OptixProgramGroupDesc prog_desc = {};
		prog_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		prog_desc.raygen.module = module;
		prog_desc.raygen.entryFunctionName = "__raygen__perspective_camera";

		log_size = sizeof(log);
		CHECK_OPTIX(optixProgramGroupCreate(optix_context, &prog_desc, 1, &prog_opts,
					log, &log_size, &raygen_prog));
		if (log_size > 0) {
			std::cout << log << "\n";
		}
	}

	// Make the miss shader programs, one for each ray type
	std::array<OptixProgramGroup, 2> miss_progs;
	{
		std::array<OptixProgramGroupDesc, 2> prog_desc = {};
		for (auto &g : prog_desc) {
			g.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			g.miss.module = module;
		}
		prog_desc[0].miss.entryFunctionName = "__miss__miss";
		prog_desc[1].miss.entryFunctionName = "__miss__occlusion_miss";

		log_size = sizeof(log);
		CHECK_OPTIX(optixProgramGroupCreate(optix_context, prog_desc.data(), prog_desc.size(),
					&prog_opts, log, &log_size, miss_progs.data()));
		if (log_size > 0) {
			std::cout << log << "\n";
		}
	}

	// Make the hit groups, for each ray type
	std::array<OptixProgramGroup, 2> hitgroup_progs;
	{
		std::array<OptixProgramGroupDesc, 2> prog_desc = {};
		for (auto &g : prog_desc) {
			g.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			g.hitgroup.moduleCH = module;
		}
		prog_desc[0].hitgroup.entryFunctionNameCH = "__closesthit__closest_hit";
		prog_desc[1].hitgroup.entryFunctionNameCH = "__closesthit__occlusion_hit";

		log_size = sizeof(log);
		CHECK_OPTIX(optixProgramGroupCreate(optix_context, prog_desc.data(), prog_desc.size(),
					&prog_opts, log, &log_size, hitgroup_progs.data()));
		if (log_size > 0) {
			std::cout << log << "\n";
		}
	}
	
	// Combine the programs into a pipeline
	{
		std::array<OptixProgramGroup, 5> pipeline_progs = {
			raygen_prog, miss_progs[0], miss_progs[1], hitgroup_progs[0], hitgroup_progs[1]
		};


		OptixPipelineLinkOptions link_opts = {};
		link_opts.maxTraceDepth = 1;
		// TODO pick debug level based on compile config
		link_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

		log_size = sizeof(log);
		CHECK_OPTIX(optixPipelineCreate(optix_context, &pipeline_opts, &link_opts,
					pipeline_progs.data(), pipeline_progs.size(),
					log, &log_size, &pipeline));
		if (log_size > 0) {
			std::cout << log << "\n";
		}
	}

	// TODO: Compute a tight bound on the stack size we need.
	// Since the path tracer is iterative, we should only need a very small stack,
	// likely smaller than the default estimate.
	// In the renderer, the raygen will call the closest hit or miss shader, which
	// make no further calls.
	{
		OptixStackSizes stack_sizes;
		optixProgramGroupGetStackSize(raygen_prog, &stack_sizes);
		std::cout << "RayGen: " << stack_sizes << "\n";

		for (size_t i = 0; i < miss_progs.size(); ++i) {
			optixProgramGroupGetStackSize(miss_progs[i], &stack_sizes);
			std::cout << "Miss[" << i << "]: " << stack_sizes << "\n";
		}
		for (size_t i = 0; i < hitgroup_progs.size(); ++i) {
			optixProgramGroupGetStackSize(hitgroup_progs[i], &stack_sizes);
			std::cout << "HitGroup[" << i << "]: " << stack_sizes << "\n";
		}

		// TODO: It seems like even setting these values to something clearly too small
		// doesn't crash the renderer like I'd expect it too?

		CHECK_OPTIX(optixPipelineSetStackSize(pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 2));
	}

	const size_t raygen_entry_size = align_to(sizeof(RayGenParams), OPTIX_SBT_RECORD_ALIGNMENT);
	const size_t miss_entry_size = align_to(sizeof(MissParams), OPTIX_SBT_RECORD_ALIGNMENT);
	const size_t hitgroup_entry_size = align_to(sizeof(HitGroupParams), OPTIX_SBT_RECORD_ALIGNMENT);
	const size_t sbt_size = raygen_entry_size + 2 * miss_entry_size + 2 * hitgroup_entry_size;
	std::cout << "SBT size: " << pretty_print_count(sbt_size) << "\n"
		<< "raygen size: " << raygen_entry_size << "\n"
		<< "miss size: " << miss_entry_size << "\n"
		<< "hit size: " << hitgroup_entry_size << "\n";

	std::vector<uint8_t> host_sbt(sbt_size, 0);
	uint8_t *sbt_ptr = host_sbt.data();
	{
		RayGenParams rg_rec;
		optixSbtRecordPackHeader(raygen_prog, &rg_rec);
		rg_rec.mat_params = reinterpret_cast<CUdeviceptr>(mat_params);

		std::memcpy(sbt_ptr, &rg_rec, sizeof(RayGenParams));
		sbt_ptr += raygen_entry_size;
	}

	for (size_t i = 0; i < miss_progs.size(); ++i) {
		MissParams miss_rec;
		optixSbtRecordPackHeader(miss_progs[i], &miss_rec);

		std::memcpy(sbt_ptr, &miss_rec, sizeof(MissParams));
		sbt_ptr += miss_entry_size;
	}

	for (size_t i = 0; i < hitgroup_progs.size(); ++i) {
		HitGroupParams hit_rec;
		optixSbtRecordPackHeader(hitgroup_progs[i], &hit_rec);
		hit_rec.vertex_buffer = reinterpret_cast<CUdeviceptr>(vertices);
		hit_rec.index_buffer = reinterpret_cast<CUdeviceptr>(indices);

		std::memcpy(sbt_ptr, &hit_rec, sizeof(HitGroupParams));
		sbt_ptr += hitgroup_entry_size;
	}

	cudaMalloc(&shader_table_data, sbt_size);
	CHECK_CUDA(cudaMemcpy(shader_table_data, host_sbt.data(), sbt_size, cudaMemcpyHostToDevice));

	std::memset(&shader_table, 0, sizeof(OptixShaderBindingTable));
	shader_table.raygenRecord = reinterpret_cast<CUdeviceptr>(shader_table_data);

	shader_table.missRecordBase = shader_table.raygenRecord + raygen_entry_size;
	shader_table.missRecordStrideInBytes = miss_entry_size;
	shader_table.missRecordCount = 2;

	shader_table.hitgroupRecordBase = shader_table.missRecordBase + 2 * miss_entry_size;
	shader_table.hitgroupRecordStrideInBytes = hitgroup_entry_size;
	shader_table.hitgroupRecordCount = 2;
}

double RenderOptiX::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;

	if (camera_changed) {
		frame_id = 0;
	}

	update_view_parameters(pos, dir, up, fovy);

	auto start = high_resolution_clock::now();

	CHECK_OPTIX(optixLaunch(pipeline, cuda_stream,
			reinterpret_cast<CUdeviceptr>(launch_params), sizeof(LaunchParams),
			&shader_table, width, height, 1));

	// Sync with the GPU to ensure it actually finishes rendering
	sync_gpu();
	auto end = high_resolution_clock::now();

	const double render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-9;

	cudaMemcpy(img.data(), framebuffer, img.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	++frame_id;
	return img.size() / render_time;
}

void RenderOptiX::update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
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
	params.framebuffer = reinterpret_cast<CUdeviceptr>(framebuffer);
	params.accum_buffer = reinterpret_cast<CUdeviceptr>(accum_buffer);
	params.scene = tlas_handle;

	cudaMemcpy(launch_params, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice);
}

void RenderOptiX::sync_gpu() {
	cudaDeviceSynchronize();
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "CUDA error " << cudaGetErrorName(err) << ": "
			<< cudaGetErrorString(err) << std::endl << std::flush;
		throw std::runtime_error("sync");
	}
}

