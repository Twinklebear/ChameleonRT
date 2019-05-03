#include <iostream>
#include "render_dxr.h"

using Microsoft::WRL::ComPtr;

RenderDXR::RenderDXR() {
	// Enable debugging for D3D12
	ComPtr<ID3D12Debug> debug_controller;
	auto err = D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
	if (FAILED(err)) {
		std::cout << "Failed to enable debug layer!\n";
		throw std::runtime_error("get debug failed");
	}
	debug_controller->EnableDebugLayer();

	/*
	ComPtr<IDXGIFactory4> factory;
	err = CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory));
	if (FAILED(err)) {
		std::cout << "Failed to make DXGI factory\n";
		throw std::runtime_error("create factor failed");
	}

	ComPtr<IDXGIAdapter1> gpu_adapter;
	GetHardwareAdapter(factory.Get(), &gpu_adapter);
	*/

	err = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
	if (FAILED(err)) {
		std::cout << "Failed to make D3D12 device\n";
		throw std::runtime_error("failed to make d3d12 device\n");
	}
}

void RenderDXR::initialize(const int fb_width, const int fb_height) {
	img.resize(fb_width * fb_height);
}
void RenderDXR::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
}
double RenderDXR::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	std::fill(img.begin(), img.end(), 255 << 16);
	return 1.f;
}

