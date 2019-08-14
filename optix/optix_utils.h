#pragma once

#include <memory>
#include <vector>
#include <array>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <glm/glm.hpp>

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

namespace optix {

class Buffer {
	size_t buf_size = 0;
	void *ptr = nullptr;

public:
	Buffer() = default;
	Buffer(size_t size);
	~Buffer();

	Buffer(const Buffer &) = delete;
	Buffer& operator=(const Buffer &) = delete;

	Buffer(Buffer &&b);
	Buffer& operator=(Buffer &&b);

	CUdeviceptr device_ptr() const;

	size_t size() const;

	void upload(const void *data, size_t size);

	void download(void *data, size_t size);

	template<typename T>
	void upload(const std::vector<T> &data);

	template<typename T, size_t N>
	void upload(const std::array<T, N> &data);

	template<typename T>
	void download(std::vector<T> &data);

	template<typename T, size_t N>
	void download(std::array<T, N> &data);

	void clear();
};

template<typename T>
void Buffer::upload(const std::vector<T> &data) {
	upload(data.data(), data.size() * sizeof(T));
}

template<typename T, size_t N>
void Buffer::upload(const std::array<T, N> &data) {
	upload(data.data(), data.size() * sizeof(T));
}

template<typename T>
void Buffer::download(std::vector<T> &data) {
	download(data.data(), data.size() * sizeof(T));
}

template<typename T, size_t N>
void Buffer::download(std::array<T, N> &data) {
	download(data.data(), data.size() * sizeof(T));
}

class TriangleMesh {
	OptixBuildInput geom_desc = {};
	CUdeviceptr vertex_ptr;

	uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_NONE;
	uint32_t build_flags = OPTIX_BUILD_FLAG_NONE;

	Buffer build_output, scratch, post_build_info, bvh;

	OptixTraversableHandle as_handle;

public:
	std::shared_ptr<Buffer> vertex_buf, index_buf, normal_buf, uv_buf;

	TriangleMesh() = default;
	// TODO: Allow other vertex and index formats? Right now this
	// assumes vec3f verts and uint3 indices
	TriangleMesh(std::shared_ptr<Buffer> vertex_buf, std::shared_ptr<Buffer> index_buf,
			std::shared_ptr<Buffer> normal_buf, std::shared_ptr<Buffer> uv_buf,
			uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_NONE,
			uint32_t build_flags = OPTIX_BUILD_FLAG_NONE);

	// Enqueue the acceleration structure build construction into the passed stream
	void enqueue_build(OptixDeviceContext &device, CUstream &stream);

	// Enqueue the acceleration structure compaction into the passed stream
	void enqueue_compaction(OptixDeviceContext &device, CUstream &stream);

	// Finalize the BVH build structures to release any scratch space
	void finalize();

	size_t num_tris() const;

	OptixTraversableHandle handle();
};

}

