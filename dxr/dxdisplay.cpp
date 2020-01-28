#include "dxdisplay.h"
#include <codecvt>
#include <locale>
#include <SDL_syswm.h>
#include "display/imgui_impl_sdl.h"
#include "imgui_impl_dx12.h"
#include "util.h"

using Microsoft::WRL::ComPtr;

DXDisplay::DXDisplay(SDL_Window *window)
{
    SDL_SysWMinfo wm_info;
    SDL_VERSION(&wm_info.version);
    SDL_GetWindowWMInfo(window, &wm_info);
    win_handle = wm_info.info.win.window;

    // Enable debugging for D3D12
#ifdef _DEBUG
    {
        ComPtr<ID3D12Debug> debug_controller;
        auto err = D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
        if (FAILED(err)) {
            std::cout << "Failed to enable debug layer!\n";
            throw std::runtime_error("get debug failed");
        }
        debug_controller->EnableDebugLayer();
    }
#endif

#ifdef _DEBUG
    uint32_t factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#else
    uint32_t factory_flags = 0;
#endif
    CHECK_ERR(CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&factory)));

    // TODO: we should enumerate the devices and find the first one supporting RTX

    auto err = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
    if (FAILED(err)) {
        std::cout << "Failed to make D3D12 device\n";
        throw std::runtime_error("failed to make d3d12 device\n");
    }

    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fence_evt = CreateEvent(nullptr, false, false, nullptr);

    // Create the command queue and command allocator
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK_ERR(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&cmd_queue)));
    CHECK_ERR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&cmd_allocator)));

    CHECK_ERR(device->CreateCommandList(0,
                                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        cmd_allocator.Get(),
                                        nullptr,
                                        IID_PPV_ARGS(&cmd_list)));

    CHECK_ERR(cmd_list->Close());

    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fence_evt = CreateEvent(nullptr, false, false, nullptr);
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {0};
        desc.NumDescriptors = render_targets.size();
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CHECK_ERR(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&render_target_desc_heap)));
    }
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {0};
        desc.NumDescriptors = 1;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_ERR(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&imgui_desc_heap)));
    }

    ImGui_ImplSDL2_InitForD3D(window);
    ImGui_ImplDX12_Init(device.Get(),
                        1,
                        DXGI_FORMAT_R8G8B8A8_UNORM,
                        imgui_desc_heap.Get(),
                        imgui_desc_heap->GetCPUDescriptorHandleForHeapStart(),
                        imgui_desc_heap->GetGPUDescriptorHandleForHeapStart());
}

DXDisplay::~DXDisplay()
{
    ImGui_ImplDX12_Shutdown();
}

std::string DXDisplay::gpu_brand()
{
    IDXGIAdapter1 *adapter;
    factory->EnumAdapters1(0, &adapter);
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(desc.Description);
}

std::string DXDisplay::name()
{
    return "DirectX 12";
}

void DXDisplay::resize(const int fb_width, const int fb_height)
{
    fb_dims = glm::uvec2(fb_width, fb_height);

    upload_texture = dxr::Buffer::upload(
        device.Get(), fb_linear_row_pitch() * fb_dims.y, D3D12_RESOURCE_STATE_GENERIC_READ);

    if (!swap_chain) {
        // Describe and create the swap chain.
        DXGI_SWAP_CHAIN_DESC1 swap_chain_desc = {0};
        swap_chain_desc.BufferCount = 2;
        swap_chain_desc.Width = fb_dims.x;
        swap_chain_desc.Height = fb_dims.y;
        swap_chain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swap_chain_desc.SampleDesc.Count = 1;

        ComPtr<IDXGISwapChain1> sc;
        CHECK_ERR(factory->CreateSwapChainForHwnd(
            cmd_queue.Get(), win_handle, &swap_chain_desc, nullptr, nullptr, &sc));

        CHECK_ERR(sc.As(&swap_chain));
    } else {
        // If the swap chain already exists, resize it
        CHECK_ERR(
            swap_chain->ResizeBuffers(2, fb_dims.x, fb_dims.y, DXGI_FORMAT_R8G8B8A8_UNORM, 0));
    }

    const uint32_t rtv_descriptor_size =
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    for (size_t i = 0; i < render_targets.size(); ++i) {
        std::memset(&render_targets[i], 0, sizeof(D3D12_CPU_DESCRIPTOR_HANDLE));
        render_targets[i] = render_target_desc_heap->GetCPUDescriptorHandleForHeapStart();
        render_targets[i].ptr += i * rtv_descriptor_size;

        ComPtr<ID3D12Resource> target;
        CHECK_ERR(swap_chain->GetBuffer(i, IID_PPV_ARGS(&target)));
        device->CreateRenderTargetView(target.Get(), nullptr, render_targets[i]);
    }
}

void DXDisplay::new_frame()
{
    ImGui_ImplDX12_NewFrame();
}

void DXDisplay::display(const std::vector<uint32_t> &img)
{
    // TODO: A utility for uploading these strided buffers for texture copies
    if (fb_linear_row_pitch() == img.size() * sizeof(uint32_t)) {
        std::memcpy(upload_texture.map(), img.data(), upload_texture.size());
    } else {
        uint8_t *buf = static_cast<uint8_t *>(upload_texture.map());
        for (uint32_t y = 0; y < fb_dims.y; ++y) {
            std::memcpy(buf + y * fb_linear_row_pitch(),
                        img.data() + y * fb_dims.x,
                        fb_dims.x * sizeof(uint32_t));
        }
    }
    upload_texture.unmap();

    CHECK_ERR(cmd_allocator->Reset());
    CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

    const uint32_t back_buffer_idx = swap_chain->GetCurrentBackBufferIndex();
    ComPtr<ID3D12Resource> back_buffer;
    CHECK_ERR(swap_chain->GetBuffer(back_buffer_idx, IID_PPV_ARGS(&back_buffer)));

    auto b = dxr::barrier_transition(
        back_buffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_list->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION dst_desc = {0};
    dst_desc.pResource = back_buffer.Get();
    dst_desc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_desc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src_desc = {0};
    src_desc.pResource = upload_texture.get();
    src_desc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_desc.PlacedFootprint.Offset = 0;
    src_desc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src_desc.PlacedFootprint.Footprint.Width = fb_dims.x;
    src_desc.PlacedFootprint.Footprint.Height = fb_dims.y;
    src_desc.PlacedFootprint.Footprint.Depth = 1;
    src_desc.PlacedFootprint.Footprint.RowPitch = fb_linear_row_pitch();

    D3D12_BOX region = {0};
    region.left = 0;
    region.right = fb_dims.x;
    region.top = 0;
    region.bottom = fb_dims.y;
    region.front = 0;
    region.back = 1;
    cmd_list->CopyTextureRegion(&dst_desc, 0, 0, 0, &src_desc, &region);

    b = dxr::barrier_transition(
        back_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd_list->ResourceBarrier(1, &b);

    // Render ImGui to the framebuffer
    cmd_list->OMSetRenderTargets(1, &render_targets[back_buffer_idx], false, nullptr);
    ID3D12DescriptorHeap *desc_heap = imgui_desc_heap.Get();
    cmd_list->SetDescriptorHeaps(1, &desc_heap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.Get());

    b = dxr::barrier_transition(
        back_buffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    cmd_list->ResourceBarrier(1, &b);

    CHECK_ERR(cmd_list->Close());

    // Execute the command list and present
    ID3D12CommandList *cmd_lists = cmd_list.Get();
    cmd_queue->ExecuteCommandLists(1, &cmd_lists);
    CHECK_ERR(swap_chain->Present(1, 0));

    // Sync with the fence to wait for the frame to be presented
    const uint64_t signal_val = fence_value++;
    CHECK_ERR(cmd_queue->Signal(fence.Get(), signal_val));

    if (fence->GetCompletedValue() < signal_val) {
        CHECK_ERR(fence->SetEventOnCompletion(signal_val, fence_evt));
        WaitForSingleObject(fence_evt, INFINITE);
    }
}

void DXDisplay::display_native(dxr::Texture2D &img)
{
    CHECK_ERR(cmd_allocator->Reset());
    CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

    const uint32_t back_buffer_idx = swap_chain->GetCurrentBackBufferIndex();
    ComPtr<ID3D12Resource> back_buffer;
    CHECK_ERR(swap_chain->GetBuffer(back_buffer_idx, IID_PPV_ARGS(&back_buffer)));

    {
        const std::array<D3D12_RESOURCE_BARRIER, 2> b = {
            dxr::barrier_transition(back_buffer.Get(),
                                    D3D12_RESOURCE_STATE_PRESENT,
                                    D3D12_RESOURCE_STATE_COPY_DEST),
            dxr::barrier_transition(img, D3D12_RESOURCE_STATE_COPY_SOURCE)};

        cmd_list->ResourceBarrier(b.size(), b.data());
    }

    cmd_list->CopyResource(back_buffer.Get(), img.get());

    {
        const std::array<D3D12_RESOURCE_BARRIER, 2> b = {
            dxr::barrier_transition(back_buffer.Get(),
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET),
            dxr::barrier_transition(img, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};

        cmd_list->ResourceBarrier(b.size(), b.data());
    }

    // Render ImGui to the framebuffer
    cmd_list->OMSetRenderTargets(1, &render_targets[back_buffer_idx], false, nullptr);
    ID3D12DescriptorHeap *desc_heap = imgui_desc_heap.Get();
    cmd_list->SetDescriptorHeaps(1, &desc_heap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd_list.Get());

    auto b = dxr::barrier_transition(
        back_buffer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    cmd_list->ResourceBarrier(1, &b);

    CHECK_ERR(cmd_list->Close());

    // Execute the command list and present
    ID3D12CommandList *cmd_lists = cmd_list.Get();
    cmd_queue->ExecuteCommandLists(1, &cmd_lists);
    CHECK_ERR(swap_chain->Present(1, 0));

    // Sync with the fence to wait for the frame to be presented
    const uint64_t signal_val = fence_value++;
    CHECK_ERR(cmd_queue->Signal(fence.Get(), signal_val));

    if (fence->GetCompletedValue() < signal_val) {
        CHECK_ERR(fence->SetEventOnCompletion(signal_val, fence_evt));
        WaitForSingleObject(fence_evt, INFINITE);
    }
}

size_t DXDisplay::fb_linear_row_pitch() const
{
    return align_to(fb_dims.x * sizeof(uint32_t), D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
}
