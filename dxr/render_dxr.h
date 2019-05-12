#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include "dx12_utils.h"
#include "dxr_utils.h"
#include "render_backend.h"

struct RenderDXR : RenderBackend {
	Microsoft::WRL::ComPtr<ID3D12Device5> device;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;
	
	Buffer vertex_buf, index_buf, view_param_buf, img_readback_buf,
		bottom_level_as, instance_buf, top_level_as, shader_table;

	Texture2D render_target;

	RootSignature raygen_root_sig, hitgroup_root_sig, global_root_sig;
	
	Microsoft::WRL::ComPtr<ID3D12StateObject> rt_state_object;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> raygen_shader_desc_heap,
		hitgroup_shader_desc_heap;

	uint64_t shader_table_entry_size = 0;
	
	uint64_t fence_value = 1;
	Microsoft::WRL::ComPtr<ID3D12Fence> fence;
	HANDLE fence_evt;

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
	void build_empty_global_sig();
	void build_shader_resource_heap();
	void build_shader_binding_table();
	void update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy);
	void update_descriptor_heap();
	void sync_gpu();
};
