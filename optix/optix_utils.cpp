#include <iostream>
#include "optix_utils.h"

namespace optix {

Buffer::Buffer(size_t size) : buf_size(size) {
	CHECK_CUDA(cudaMalloc(&ptr, buf_size));
}
Buffer::~Buffer() {
	cudaFree(ptr);
}

Buffer::Buffer(Buffer &&b)
	: buf_size(b.buf_size), ptr(b.ptr)
{
	b.buf_size = 0;
	b.ptr = nullptr;
}
Buffer& Buffer::operator=(Buffer &&b) {
	CHECK_CUDA(cudaFree(ptr));
	buf_size = b.buf_size;
	ptr = b.ptr;

	b.buf_size = 0;
	b.ptr = nullptr;

	return *this;
}

CUdeviceptr Buffer::device_ptr() const {
	return reinterpret_cast<CUdeviceptr>(ptr);
}

size_t Buffer::size() const {
	return buf_size;
}

void Buffer::upload(const void *data, size_t size) {
	CHECK_CUDA(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
}

void Buffer::download(void *data, size_t size) {
	CHECK_CUDA(cudaMemcpy(data, ptr, size, cudaMemcpyDeviceToHost));
}

void Buffer::clear() {
	CHECK_CUDA(cudaMemset(ptr, 0, buf_size));
}

}

