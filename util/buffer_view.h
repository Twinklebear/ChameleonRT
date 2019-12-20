#pragma once

#include "gltf_types.h"
#include "tiny_gltf.h"

// GLTF Buffer view/accessor utilities

struct BufferView {
    const uint8_t *buf = nullptr;
    size_t length = 0;
    size_t stride = 0;

    BufferView() = default;

    BufferView(const tinygltf::BufferView &view,
               const tinygltf::Model &model,
               size_t base_stride);

    BufferView(const uint8_t *buf, size_t byte_length, size_t byte_stride);

    // Return the pointer to some element, based on the stride specified for the view
    const uint8_t *operator[](const size_t i) const;
};

template <typename T>
class Accessor {
    BufferView view;
    size_t count = 0;

public:
    Accessor() = default;

    Accessor(const tinygltf::Accessor &accessor, const tinygltf::Model &model);

    Accessor(const BufferView &view);

    const T &operator[](const size_t i) const;

    // Note: begin/end require the buffer to be tightly packed
    const T *begin() const;

    const T *end() const;

    size_t size() const;
};

template <typename T>
Accessor<T>::Accessor(const tinygltf::Accessor &accessor, const tinygltf::Model &model)
    : view(model.bufferViews[accessor.bufferView],
           model,
           gltf_base_stride(accessor.type, accessor.componentType)),
      count(accessor.count)
{
    // Apply the additional accessor-specific byte offset
    view.buf += accessor.byteOffset;
}

template <typename T>
Accessor<T>::Accessor(const BufferView &view) : view(view), count(view.length / view.stride)
{
}

template <typename T>
const T *Accessor<T>::begin() const
{
    if (view.length / view.stride != count) {
        throw std::runtime_error("Accessor<T>::begin cannot be used on non-packed buffer");
    }
    return reinterpret_cast<const T *>(view[0]);
}

template <typename T>
const T *Accessor<T>::end() const
{
    if (view.length / view.stride != count) {
        throw std::runtime_error("Accessor<T>::end cannot be used on non-packed buffer");
    }
    return reinterpret_cast<const T *>(view[count]);
}

template <typename T>
const T &Accessor<T>::operator[](const size_t i) const
{
    return *reinterpret_cast<const T *>(view[i]);
}

template <typename T>
size_t Accessor<T>::size() const
{
    return count;
}
