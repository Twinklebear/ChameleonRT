#include <chrono>
#include <cuda.h>
#include "render_optix.h"

#define CHECK_OPTIX(FN) \
	{ \
		auto fn_err = FN; \
		if (fn_err != OPTIX_SUCCESS) { \
			std::cout << #FN << " failed due to " \
				<< optixGetErrorName(fn_err) << ": " << optixGetErrorString(fn_err) \
				<< std::endl << std::flush; \
			throw std::runtime_error(#FN); \
		}\
	}

#define CHECK_CUDA(FN) \
	{ \
		auto fn_err = FN; \
		if (fn_err != cudaSuccess) { \
			std::cout << #FN << " failed due to " \
				<< cudaGetErrorName(fn_err) << ": " << cudaGetErrorString(fn_err) \
				<< std::endl << std::flush; \
			throw std::runtime_error(#FN); \
		}\
	}

extern "C" const char render_optix_ptx[];

void log_callback(unsigned int level, const char *tag, const char *msg, void *data) {
	std::cout << "----\nOptiX Log Message (level " << level << "):\n"
		<< "  Tag: " << tag << "\n"
		<< "  Msg: " << msg << "\n----\n";
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

	CHECK_CUDA(cuCtxGetCurrent(&cuda_context));

	CHECK_OPTIX(optixDeviceContextCreate(cuda_context, 0, &optix_context));
#ifdef _DEBUG
	CHECK_OPTIX(optixDeviceContextSetLogCallback(optix_context, log_callback, nullptr, 4));
#else
	CHECK_OPTIX(optixDeviceContextSetLogCallback(optix_context, log_callback, nullptr, 3));
#endif

	/*
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(1);
	Program prog = context->createProgramFromPTXString(render_optix_ptx, "perspective_camera");
	context->setRayGenerationProgram(0, prog);

	view_params = context->createBuffer(RT_BUFFER_INPUT);
	view_params->setFormat(RT_FORMAT_USER);
	view_params->setElementSize(5 * sizeof(glm::vec4));
	view_params->setSize(1);
	context["view_params"]->set(view_params);

	mat_params = context->createBuffer(RT_BUFFER_INPUT);
	mat_params->setFormat(RT_FORMAT_USER);
	mat_params->setElementSize(4 * sizeof(glm::vec4));
	mat_params->setSize(1);
	context["mat_params"]->set(mat_params);
	*/
}

void RenderOptiX::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
	width = fb_width;
	height = fb_height;
	img.resize(fb_width * fb_height);

	if (fb) {
		cudaFree(fb);
		cudaFree(accum_buffer);
	}
	cudaMalloc(&fb, img.size() * sizeof(uint32_t));
	cudaMalloc(&accum_buffer, img.size() * 4 * sizeof(float));
}

void RenderOptiX::set_scene(const Scene &scene) {
	/*
	frame_id = 0;
	const auto &mesh = scene.meshes[0];

	auto vertex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
			mesh.vertices.size());
	std::copy(mesh.vertices.begin(), mesh.vertices.end(),
			static_cast<glm::vec3*>(vertex_buffer->map()));
	vertex_buffer->unmap();

	auto index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3,
			mesh.indices.size());
	std::copy(mesh.indices.begin(), mesh.indices.end(),
			static_cast<glm::uvec3*>(index_buffer->map()));
	index_buffer->unmap();

	auto geom_tri = context->createGeometryTriangles();
	geom_tri->setPrimitiveCount(mesh.indices.size());
	geom_tri->setTriangleIndices(index_buffer, RT_FORMAT_UNSIGNED_INT3);
	geom_tri->setVertices(mesh.vertices.size(), vertex_buffer, RT_FORMAT_FLOAT3);

	auto mat = context->createMaterial();
	mat->setClosestHitProgram(0,
			context->createProgramFromPTXString(render_optix_ptx, "closest_hit"));
	mat->setClosestHitProgram(1,
		context->createProgramFromPTXString(render_optix_ptx, "occlusion_hit"));

	auto instance = context->createGeometryInstance(geom_tri, mat);
	// We use these in the hit program to color by normal
	instance["index_buffer"]->set(index_buffer);
	instance["vertex_buffer"]->set(vertex_buffer);

	auto optixscene = context->createGeometryGroup();
	optixscene->addChild(instance);
	optixscene->setAcceleration(context->createAcceleration("Trbvh"));

	context["scene"]->set(optixscene);

	std::memcpy(mat_params->map(), &scene.materials[mesh.material_id], sizeof(DisneyMaterial));
	mat_params->unmap();
	*/
}

double RenderOptiX::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;
	/*

	if (camera_changed) {
		frame_id = 0;
		try {
			context->validate();
		} catch (const optix::Exception &e) {
			std::cout << "OptiX Error: " << e.getErrorString() << "\n" << std::flush;
			throw std::runtime_error(e.getErrorString());
		}
	}

	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(width) / height;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	// Work around Optix re-upload bug by updating parameters which change each frame through
	// a buffer instead of as direct parameters
	uint8_t *map = static_cast<uint8_t*>(view_params->map());
	{
		glm::vec4 *vecs = reinterpret_cast<glm::vec4*>(map);
		vecs[0] = glm::vec4(pos, 0);
		vecs[1] = glm::vec4(dir_du, 0);
		vecs[2] = glm::vec4(dir_dv, 0);
		vecs[3] = glm::vec4(dir_top_left, 0);
	}
	{
		uint32_t *fid = reinterpret_cast<uint32_t*>(map + 4 * sizeof(glm::vec4));
		*fid = frame_id;
	}
	view_params->unmap();
	
	auto start = high_resolution_clock::now();

	context->launch(0, width, height);
	// Sync with the GPU to ensure it actually finishes rendering
	cudaDeviceSynchronize();
	auto end = high_resolution_clock::now();
	const double render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-9;

	const uint32_t *mapped = static_cast<const uint32_t*>(fb->map(0, RT_BUFFER_MAP_READ));
	std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
	fb->unmap();
	++frame_id;
	return img.size() / render_time;
	*/
	return 0;
}

