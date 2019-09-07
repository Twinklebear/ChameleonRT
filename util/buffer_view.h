#pragma once

#include "gltf_types.h"
#include "tiny_gltf.h"

// GLTF Buffer view/accessor utilities

struct BufferView {
    const uint8_t *buf = nullptr;
    size_t length = 0;
    size_t stride = 0;

    BufferView(const tinygltf::BufferView &view, const tinygltf::Model &model, size_t base_stride);

    BufferView() = default;

    // Return the pointer to some element, based on the stride specified for the view
    const uint8_t *operator[](const size_t i) const;
};

template <typename T>
class Accessor {
    BufferView view;
    size_t count = 0;

public:
    Accessor(const tinygltf::Accessor &accessor, const tinygltf::Model &model);

    Accessor() = default;

    const T &operator[](const size_t i) const;

    const size_t size() const;
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
const T &Accessor<T>::operator[](const size_t i) const
{
    return *reinterpret_cast<const T *>(view[i]);
}

template <typename T>
const size_t Accessor<T>::size() const
{
    return count;
}
