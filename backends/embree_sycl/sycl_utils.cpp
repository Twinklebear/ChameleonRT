#include "sycl_utils.h"
#include <stdexcept>

namespace embree {

Buffer::Buffer(size_t size, MemorySpace mem_space, sycl::queue &sycl_queue)
    : buf_size(size), memory_space(mem_space), squeue(&sycl_queue)
{
    switch (mem_space) {
    case MemorySpace::SHARED:
        ptr = sycl::malloc_shared(buf_size, sycl_queue);
        break;
    case MemorySpace::HOST:
        ptr = sycl::malloc_host(buf_size, sycl_queue);
        break;
    case MemorySpace::DEVICE:
        ptr = sycl::malloc_device(buf_size, sycl_queue);
        break;
    }
}
Buffer::~Buffer()
{
    sycl::free(ptr, *squeue);
}

Buffer::Buffer(Buffer &&b)
    : buf_size(b.buf_size), memory_space(b.memory_space), ptr(b.ptr), squeue(b.squeue)
{
    b.buf_size = 0;
    b.ptr = nullptr;
}
Buffer &Buffer::operator=(Buffer &&b)
{
    if (ptr) {
        sycl::free(ptr, *squeue);
    }
    buf_size = b.buf_size;
    memory_space = b.memory_space;
    ptr = b.ptr;
    squeue = b.squeue;

    b.buf_size = 0;
    b.ptr = nullptr;
    b.squeue = nullptr;

    return *this;
}

void *Buffer::device_ptr() const
{
    if (memory_space == MemorySpace::HOST) {
        throw std::runtime_error("Request for device_ptr on non-device accessible Buffer");
    }
    return ptr;
}

void *Buffer::host_ptr() const
{
    if (memory_space == MemorySpace::DEVICE) {
        throw std::runtime_error("Request for host_ptr on non-host accessible Buffer");
    }
    return ptr;
}

size_t Buffer::size() const
{
    return buf_size;
}

sycl::event Buffer::upload(const void *data, size_t size, sycl::queue &sycl_queue)
{
    return sycl_queue.memcpy(ptr, data, size);
}

sycl::event Buffer::download(void *data, size_t size, sycl::queue &sycl_queue)
{
    return sycl_queue.memcpy(data, ptr, size);
}

sycl::event Buffer::clear(sycl::queue &sycl_queue)
{
    return sycl_queue.memset(ptr, 0, buf_size);
}

}
