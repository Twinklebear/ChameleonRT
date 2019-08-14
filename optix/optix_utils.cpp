#include <iostream>
#include <optix_stubs.h>
#include "util.h"
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

TriangleMesh::TriangleMesh(std::shared_ptr<Buffer> vertex_buf, std::shared_ptr<Buffer> index_buf,
		std::shared_ptr<Buffer> normal_buf, std::shared_ptr<Buffer> uv_buf,
		uint32_t g_flags, uint32_t build_flags)
	: vertex_ptr(vertex_buf->device_ptr()),
	geom_flags(g_flags),
	build_flags(build_flags),
	vertex_buf(vertex_buf),
	index_buf(index_buf),
	normal_buf(normal_buf),
	uv_buf(uv_buf)
{
	geom_desc.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	// TODO: Support for multiple geom in a single bottom level BVH
	geom_desc.triangleArray.vertexBuffers = &vertex_ptr;
	geom_desc.triangleArray.numVertices = vertex_buf->size() / sizeof(glm::vec3);
	geom_desc.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	geom_desc.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);

	geom_desc.triangleArray.indexBuffer = index_buf->device_ptr();
	geom_desc.triangleArray.numIndexTriplets = index_buf->size() / sizeof(glm::uvec3);
	geom_desc.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	geom_desc.triangleArray.indexStrideInBytes = sizeof(glm::uvec3);

	geom_desc.triangleArray.flags = &geom_flags;
	geom_desc.triangleArray.numSbtRecords = 1;
}

void TriangleMesh::enqueue_build(OptixDeviceContext &device, CUstream &stream) {
	OptixAccelBuildOptions opts = {};
	opts.buildFlags =  build_flags;
	opts.operation = OPTIX_BUILD_OPERATION_BUILD;
	opts.motionOptions.numKeys = 1;

	OptixAccelBufferSizes buf_sizes;
	CHECK_OPTIX(optixAccelComputeMemoryUsage(device, &opts, &geom_desc, 1, &buf_sizes));

	std::cout << "BLAS will use output space of "
		<< pretty_print_count(buf_sizes.outputSizeInBytes)
		<< " plus scratch of " << pretty_print_count(buf_sizes.tempSizeInBytes) << "\n";

	build_output = Buffer(buf_sizes.outputSizeInBytes);
	scratch = Buffer(buf_sizes.tempSizeInBytes);

	// Now build the BLAS and query the info about the compacted size
	post_build_info = Buffer(sizeof(uint64_t));
	OptixAccelEmitDesc emit_desc = {};
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = post_build_info.device_ptr();

	CHECK_OPTIX(optixAccelBuild(device, stream, &opts, &geom_desc, 1,
				scratch.device_ptr(), scratch.size(),
				build_output.device_ptr(), build_output.size(),
				&as_handle, &emit_desc, 1));
}

void TriangleMesh::enqueue_compaction(OptixDeviceContext &device, CUstream &stream) {
	uint64_t compacted_size = 0;
	post_build_info.download(&compacted_size, sizeof(uint64_t));

	std::cout << "BLAS will compact to " << pretty_print_count(compacted_size) << "\n";
	bvh = optix::Buffer(compacted_size);

	CHECK_OPTIX(optixAccelCompact(device, stream, as_handle,
				bvh.device_ptr(), bvh.size(), &as_handle));
}

void TriangleMesh::finalize() {
	if (build_flags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
		build_output = Buffer();
	}
	scratch = Buffer();
	post_build_info = Buffer();
}

size_t TriangleMesh::num_tris() const {
	return geom_desc.triangleArray.numIndexTriplets;
}

OptixTraversableHandle TriangleMesh::handle() {
	return as_handle;
}

}

