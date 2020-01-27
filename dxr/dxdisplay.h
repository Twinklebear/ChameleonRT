#pragma once

#include <array>
#include <string>
#include <vector>
#include <SDL.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include "display/display.h"
#include "dx12_utils.h"
#include <glm/glm.hpp>

struct DXDisplay : Display {
    HWND win_handle;

    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> render_target_desc_heap, imgui_desc_heap;
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 2> render_targets;

    glm::uvec2 fb_dims;
    dxr::Buffer upload_texture;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swap_chain;

    uint64_t fence_value = 1;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE fence_evt;

    DXDisplay(SDL_Window *window);

    ~DXDisplay();

    std::string gpu_brand() override;

    std::string name() override;

    void resize(const int fb_width, const int fb_height) override;

    void new_frame() override;

    void display(const std::vector<uint32_t> &img) override;

    void display_native(dxr::Texture2D &img);

private:
    size_t fb_linear_row_pitch() const;
};
