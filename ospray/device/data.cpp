#include "data.h"
#include <ospray/SDK/common/OSPCommon.h>

using namespace ospcommon::math;

namespace device {

Data::Data(const ospcommon::math::vec3ul &dims, OSPDataType type) : dims(dims), type(type) {}

vec3ul Data::size() const
{
    return dims;
}

vec3ul Data::stride() const
{
    const size_t type_size = ospray::sizeOf(type);
    return vec3ul(type_size, type_size * dims.x, type_size * dims.x * dims.y);
}

size_t Data::size_bytes() const
{
    const size_t type_size = ospray::sizeOf(type);
    return type_size * dims.x * dims.y * dims.z;
}

uint8_t *Data::begin()
{
    return reinterpret_cast<uint8_t *>(data());
}

uint8_t *Data::end()
{
    return begin() + size_bytes();
}

const uint8_t *Data::begin() const
{
    return reinterpret_cast<const uint8_t *>(data());
}

const uint8_t *Data::end() const
{
    return begin() + size_bytes();
}

BorrowedData::BorrowedData(uint8_t *buf, const ospcommon::math::vec3ul &dims, OSPDataType type)
    : Data(dims, type), buf(buf)
{
}

uint8_t *BorrowedData::data()
{
    return buf;
}

const uint8_t *BorrowedData::data() const
{
    return buf;
}

OwnedData::OwnedData(uint8_t *in_buf, const ospcommon::math::vec3ul &dims, OSPDataType type)
    : Data(dims, type), buf(in_buf, in_buf + ospray::sizeOf(type) * dims.x * dims.y * dims.z)
{
}

OwnedData::OwnedData(const ospcommon::math::vec3ul &dims, OSPDataType type)
    : Data(dims, type), buf(dims.x * dims.y * dims.z * ospray::sizeOf(type), 0)
{
}

uint8_t *OwnedData::data()
{
    return buf.data();
}

const uint8_t *OwnedData::data() const
{
    return buf.data();
}
}
