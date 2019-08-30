#pragma once

#include <iostream>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include <glm/glm.hpp>

// Utilities for general DX12 ease of use

#define CHECK_ERR(FN)                                                                \
    {                                                                                \
        auto fn_err = FN;                                                            \
        if (FAILED(fn_err)) {                                                        \
            std::cout << #FN << " failed due to " << std::hex << fn_err << std::endl \
                      << std::flush;                                                 \
            throw std::runtime_error(#FN);                                           \
        }                                                                            \
    }

namespace dxr {

extern const D3D12_HEAP_PROPERTIES UPLOAD_HEAP_PROPS;
extern const D3D12_HEAP_PROPERTIES DEFAULT_HEAP_PROPS;
extern const D3D12_HEAP_PROPERTIES READBACK_HEAP_PROPS;

// Convenience for making resource transition barriers
D3D12_RESOURCE_BARRIER barrier_transition(ID3D12Resource *res,
                                          D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after);
D3D12_RESOURCE_BARRIER barrier_transition(Microsoft::WRL::ComPtr<ID3D12Resource> &res,
                                          D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after);

// Convenience for making UAV transition barriers
D3D12_RESOURCE_BARRIER barrier_uav(ID3D12Resource *res);
D3D12_RESOURCE_BARRIER barrier_uav(Microsoft::WRL::ComPtr<ID3D12Resource> &res);

class Resource {
protected:
    Microsoft::WRL::ComPtr<ID3D12Resource> res = nullptr;
    D3D12_HEAP_TYPE rheap;
    D3D12_RESOURCE_STATES rstate;

    friend D3D12_RESOURCE_BARRIER barrier_transition(Resource &res, D3D12_RESOURCE_STATES after);

public:
    virtual ~Resource();

    ID3D12Resource *operator->();
    const ID3D12Resource *operator->() const;
    ID3D12Resource *get();
    const ID3D12Resource *get() const;
    D3D12_HEAP_TYPE heap() const;
    D3D12_RESOURCE_STATES state() const;
};

D3D12_RESOURCE_BARRIER barrier_transition(Resource &res, D3D12_RESOURCE_STATES after);
D3D12_RESOURCE_BARRIER barrier_uav(Resource &res);

class Buffer : public Resource {
    size_t buf_size = 0;

    static D3D12_RESOURCE_DESC res_desc(size_t nbytes, D3D12_RESOURCE_FLAGS flags);

    static Buffer create(ID3D12Device *device,
                         size_t nbytes,
                         D3D12_RESOURCE_STATES state,
                         D3D12_HEAP_PROPERTIES props,
                         D3D12_RESOURCE_DESC desc);

public:
    // Allocate an upload buffer of the desired size
    static Buffer upload(ID3D12Device *device,
                         size_t nbytes,
                         D3D12_RESOURCE_STATES state,
                         D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
    // Allocate a GPU-side buffer of the desired size
    static Buffer default(ID3D12Device *device,
                          size_t nbytes,
                          D3D12_RESOURCE_STATES state,
                          D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
    // Allocate a readback buffer of the desired size
    static Buffer readback(ID3D12Device *device,
                           size_t nbytes,
                           D3D12_RESOURCE_STATES state,
                           D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);

    // Map the whole range for potentially being read
    void *map();
    // Map to read a specific or empty range
    void *map(D3D12_RANGE read);

    // Unmap and mark the whole range as written
    void unmap();
    // Unmap and mark a specific range as written
    void unmap(D3D12_RANGE written);

    size_t size() const;
};

class Texture2D : public Resource {
    glm::uvec2 tdims = glm::uvec2(0);
    DXGI_FORMAT format;

public:
    static Texture2D default(ID3D12Device *device,
                             glm::uvec2 dims,
                             D3D12_RESOURCE_STATES state,
                             DXGI_FORMAT img_format,
                             D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);

    // Read the texture data back into the provided buffer
    // buffer size must be aligned to a row pitch of D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    void readback(ID3D12GraphicsCommandList4 *cmd_list, Buffer &buf);

    // Upload the buffer into the texture
    // buffer size must be aligned to a row pitch of D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    void upload(ID3D12GraphicsCommandList4 *cmd_list, Buffer &buf);

    size_t linear_row_pitch() const;
    // Size of one pixel, in bytes
    size_t pixel_size() const;
    DXGI_FORMAT pixel_format() const;
    glm::uvec2 dims() const;
};

}
