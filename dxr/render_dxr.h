#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <wrl.h>
#include "render_backend.h"

struct RenderDXR : RenderBackend {
	Microsoft::WRL::ComPtr<ID3D12Device5> device;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;
	
	Microsoft::WRL::ComPtr<ID3D12Resource> img_readback_buf, render_target,
		vertex_buf, index_buf, bottom_level_as, top_level_as, instance_buf,
		shader_table, view_param_buf;
	
	Microsoft::WRL::ComPtr<ID3D12StateObject> rt_state_object;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> raygen_shader_desc_heap,
		hitgroup_shader_desc_heap;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> raygen_root_sig,
		hitgroup_root_sig;
	
	uint64_t fence_value = 1;
	Microsoft::WRL::ComPtr<ID3D12Fence> fence;
	HANDLE fence_evt;

	glm::ivec2 img_dims = glm::ivec2(0);
	uint32_t n_verts, n_indices;

	RenderDXR();
	virtual ~RenderDXR();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy) override;

private:
	void build_raytracing_pipeline();
	void build_raygen_root_signature();
	void build_hitgroup_root_signature();
	void build_shader_resource_heap();
	void build_shader_binding_table();
	void update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy);
	void update_descriptor_heap();
	void sync_gpu();
};
