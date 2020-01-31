#pragma once

#include <vector>
#include <ospcommon/math/vec.h>
#include <ospray/OSPEnums.h>

namespace device {

class Data {
protected:
    ospcommon::math::vec3ul dims = ospcommon::math::vec3ul(0);
    OSPDataType type = OSP_UNKNOWN;

public:
    Data(const ospcommon::math::vec3ul &dims, OSPDataType type);
    Data() = default;

    ospcommon::math::vec3ul size() const;

    ospcommon::math::vec3ul stride() const;

    size_t size_bytes() const;

    virtual uint8_t *data() = 0;

    virtual const uint8_t *data() const = 0;

    uint8_t *begin();

    uint8_t *end();

    const uint8_t *begin() const;

    const uint8_t *end() const;
};

class BorrowedData : public Data {
    uint8_t *buf = nullptr;

public:
    BorrowedData(uint8_t *buf, const ospcommon::math::vec3ul &dims, OSPDataType type);
    BorrowedData() = default;

    uint8_t *data() override;

    const uint8_t *data() const override;
};

class OwnedData : public Data {
    std::vector<uint8_t> buf;

public:
    // Copy the data into the owned buffer
    OwnedData(uint8_t *buf, const ospcommon::math::vec3ul &dims, OSPDataType type);
    OwnedData(const ospcommon::math::vec3ul &dims, OSPDataType type);
    OwnedData() = default;

    uint8_t *data() override;

    const uint8_t *data() const override;
};
}
