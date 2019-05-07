#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <wrl.h>
#include "render_backend.h"

struct RenderDXR : RenderBackend {
	Microsoft::WRL::ComPtr<ID3D12Device5> device;

	RenderDXR();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy) override;
};

