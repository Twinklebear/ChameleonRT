#pragma once

#include <CL/sycl.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace embree {

enum class MemorySpace { SHARED, HOST, DEVICE };

class Buffer {
    size_t buf_size = 0;
    MemorySpace memory_space = MemorySpace::SHARED;
    void *ptr = nullptr;
    sycl::queue *squeue;

public:
    Buffer() = default;
    Buffer(size_t size, MemorySpace memory_space, sycl::queue &sycl_queue);
    ~Buffer();

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&b);
    Buffer &operator=(Buffer &&b);

    void *device_ptr() const;

    void *host_ptr() const;

    size_t size() const;

    sycl::event upload(const void *data, size_t size, sycl::queue &sycl_queue);

    sycl::event download(void *data, size_t size, sycl::queue &sycl_queue);

    template <typename T>
    sycl::event upload(const std::vector<T> &data, sycl::queue &sycl_queue);

    template <typename T, size_t N>
    sycl::event upload(const std::array<T, N> &data, sycl::queue &sycl_queue);

    template <typename T>
    sycl::event download(std::vector<T> &data, sycl::queue &sycl_queue);

    template <typename T, size_t N>
    sycl::event download(std::array<T, N> &data, sycl::queue &sycl_queue);

    sycl::event clear(sycl::queue &sycl_queue);
};

template <typename T>
sycl::event Buffer::upload(const std::vector<T> &data, sycl::queue &sycl_queue)
{
    return upload(data.data(), data.size() * sizeof(T), sycl_queue);
}

template <typename T, size_t N>
sycl::event Buffer::upload(const std::array<T, N> &data, sycl::queue &sycl_queue)
{
    return upload(data.data(), data.size() * sizeof(T), sycl_queue);
}

template <typename T>
sycl::event Buffer::download(std::vector<T> &data, sycl::queue &sycl_queue)
{
    return download(data.data(), data.size() * sizeof(T), sycl_queue);
}

template <typename T, size_t N>
sycl::event Buffer::download(std::array<T, N> &data, sycl::queue &sycl_queue)
{
    return download(data.data(), data.size() * sizeof(T), sycl_queue);
}

}
