#include "buffer_view.h"
#include <cmath>

BufferView::BufferView(const tinygltf::BufferView &view,
                       const tinygltf::Model &model,
                       size_t base_stride)
    : buf(model.buffers[view.buffer].data.data() + view.byteOffset),
      length(view.byteLength),
      stride(std::max(view.byteStride, base_stride))
{
}

const uint8_t *BufferView::operator[](const size_t i) const
{
    return buf + i * stride;
}

