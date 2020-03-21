#include "dx12_utils.h"
#include <cassert>
#include "util.h"

namespace dxr {

using Microsoft::WRL::ComPtr;

const D3D12_HEAP_PROPERTIES UPLOAD_HEAP_PROPS = {
    D3D12_HEAP_TYPE_UPLOAD,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    0,
    0,
};

const D3D12_HEAP_PROPERTIES DEFAULT_HEAP_PROPS = {
    D3D12_HEAP_TYPE_DEFAULT,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    0,
    0,
};

const D3D12_HEAP_PROPERTIES READBACK_HEAP_PROPS = {
    D3D12_HEAP_TYPE_READBACK,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_MEMORY_POOL_UNKNOWN,
    0,
    0,
};

D3D12_RESOURCE_BARRIER barrier_transition(ID3D12Resource *res,
                                          D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.pResource = res;
    return b;
}

D3D12_RESOURCE_BARRIER barrier_transition(Microsoft::WRL::ComPtr<ID3D12Resource> &res,
                                          D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after)
{
    return barrier_transition(res.Get(), before, after);
}

D3D12_RESOURCE_BARRIER barrier_transition(Resource &res, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = barrier_transition(res.get(), res.state(), after);
    res.rstate = after;
    return b;
}

D3D12_RESOURCE_BARRIER barrier_uav(ID3D12Resource *res)
{
    D3D12_RESOURCE_BARRIER b = {0};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.UAV.pResource = res;
    return b;
}

D3D12_RESOURCE_BARRIER barrier_uav(Microsoft::WRL::ComPtr<ID3D12Resource> &res)
{
    return barrier_uav(res.Get());
}

D3D12_RESOURCE_BARRIER barrier_uav(Resource &res)
{
    return barrier_uav(res.get());
}

Resource::~Resource() {}

ID3D12Resource *Resource::operator->()
{
    return get();
}
const ID3D12Resource *Resource::operator->() const
{
    return get();
}
ID3D12Resource *Resource::get()
{
    return res.Get();
}
const ID3D12Resource *Resource::get() const
{
    return res.Get();
}
D3D12_HEAP_TYPE Resource::heap() const
{
    return rheap;
}
D3D12_RESOURCE_STATES Resource::state() const
{
    return rstate;
}

D3D12_RESOURCE_DESC Buffer::res_desc(size_t nbytes, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = nbytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

Buffer Buffer::create(ID3D12Device *device,
                      size_t nbytes,
                      D3D12_RESOURCE_STATES state,
                      D3D12_HEAP_PROPERTIES props,
                      D3D12_RESOURCE_DESC desc)
{
    if (nbytes == 0) {
        throw std::runtime_error("Error: Cannot create a buffer of size 0");
    }
    Buffer b;
    b.buf_size = nbytes;
    b.rheap = props.Type;
    b.rstate = state;
    CHECK_ERR(device->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&b.res)));
    return b;
}

Buffer Buffer::upload(ID3D12Device *device,
                      size_t nbytes,
                      D3D12_RESOURCE_STATES state,
                      D3D12_RESOURCE_FLAGS flags)
{
    return create(device, nbytes, state, UPLOAD_HEAP_PROPS, res_desc(nbytes, flags));
}
Buffer Buffer::default(ID3D12Device *device,
                       size_t nbytes,
                       D3D12_RESOURCE_STATES state,
                       D3D12_RESOURCE_FLAGS flags)
{
    return create(device, nbytes, state, DEFAULT_HEAP_PROPS, res_desc(nbytes, flags));
}
Buffer Buffer::readback(ID3D12Device *device,
                        size_t nbytes,
                        D3D12_RESOURCE_STATES state,
                        D3D12_RESOURCE_FLAGS flags)
{
    return create(device, nbytes, state, READBACK_HEAP_PROPS, res_desc(nbytes, flags));
}

void *Buffer::map()
{
    assert(rheap != D3D12_HEAP_TYPE_DEFAULT);
    void *mapping = nullptr;
    D3D12_RANGE range = {0};
    // Explicitly note we want the whole range to silence debug layer warnings
    range.End = buf_size;
    CHECK_ERR(res->Map(0, &range, &mapping));
    return mapping;
}

void *Buffer::map(D3D12_RANGE read)
{
    assert(rheap != D3D12_HEAP_TYPE_DEFAULT);
    void *mapping = nullptr;
    CHECK_ERR(res->Map(0, &read, &mapping));
    return mapping;
}

void Buffer::unmap()
{
    res->Unmap(0, nullptr);
}

void Buffer::unmap(D3D12_RANGE written)
{
    res->Unmap(0, &written);
}

size_t Buffer::size() const
{
    return buf_size;
}

Texture2D Texture2D::default(ID3D12Device *device,
                             glm::uvec2 dims,
                             D3D12_RESOURCE_STATES state,
                             DXGI_FORMAT img_format,
                             D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = dims.x;
    desc.Height = dims.y;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = img_format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    Texture2D t;
    t.tdims = dims;
    t.rstate = state;
    t.rheap = D3D12_HEAP_TYPE_DEFAULT;
    t.format = img_format;
    CHECK_ERR(device->CreateCommittedResource(&DEFAULT_HEAP_PROPS,
                                              D3D12_HEAP_FLAG_NONE,
                                              &desc,
                                              state,
                                              nullptr,
                                              IID_PPV_ARGS(&t.res)));
    return t;
}

void Texture2D::readback(ID3D12GraphicsCommandList4 *cmd_list, Buffer &buf)
{
    D3D12_TEXTURE_COPY_LOCATION dst_desc = {0};
    dst_desc.pResource = buf.get();
    dst_desc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_desc.PlacedFootprint.Offset = 0;
    dst_desc.PlacedFootprint.Footprint.Format = format;
    dst_desc.PlacedFootprint.Footprint.Width = tdims.x;
    dst_desc.PlacedFootprint.Footprint.Height = tdims.y;
    dst_desc.PlacedFootprint.Footprint.Depth = 1;
    dst_desc.PlacedFootprint.Footprint.RowPitch = linear_row_pitch();

    D3D12_TEXTURE_COPY_LOCATION src_desc = {0};
    src_desc.pResource = get();
    src_desc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_desc.SubresourceIndex = 0;

    D3D12_BOX region = {0};
    region.left = 0;
    region.right = tdims.x;
    region.top = 0;
    region.bottom = tdims.y;
    region.front = 0;
    region.back = 1;
    cmd_list->CopyTextureRegion(&dst_desc, 0, 0, 0, &src_desc, &region);
}

void Texture2D::upload(ID3D12GraphicsCommandList4 *cmd_list, Buffer &buf)
{
    D3D12_TEXTURE_COPY_LOCATION dst_desc = {0};
    dst_desc.pResource = get();
    dst_desc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_desc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src_desc = {0};
    src_desc.pResource = buf.get();
    src_desc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_desc.PlacedFootprint.Offset = 0;
    src_desc.PlacedFootprint.Footprint.Format = format;
    src_desc.PlacedFootprint.Footprint.Width = tdims.x;
    src_desc.PlacedFootprint.Footprint.Height = tdims.y;
    src_desc.PlacedFootprint.Footprint.Depth = 1;
    src_desc.PlacedFootprint.Footprint.RowPitch = linear_row_pitch();

    D3D12_BOX region = {0};
    region.left = 0;
    region.right = tdims.x;
    region.top = 0;
    region.bottom = tdims.y;
    region.front = 0;
    region.back = 1;
    cmd_list->CopyTextureRegion(&dst_desc, 0, 0, 0, &src_desc, &region);
}

size_t Texture2D::linear_row_pitch() const
{
    return align_to(tdims.x * pixel_size(), D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
}

size_t Texture2D::pixel_size() const
{
    // Just the common formats I plan to use
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return 4;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    case DXGI_FORMAT_R16_UINT:
        return 2;
    default:
        throw std::runtime_error("Unhandled format in pixel_size!");
        return -1;
    };
}

DXGI_FORMAT Texture2D::pixel_format() const
{
    return format;
}

glm::uvec2 Texture2D::dims() const
{
    return tdims;
}

}
