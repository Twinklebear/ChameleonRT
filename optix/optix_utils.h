#pragma once

#include <vector>
#include <array>
#include <cuda.h>
#include <cuda_runtime_api.h>

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

}

