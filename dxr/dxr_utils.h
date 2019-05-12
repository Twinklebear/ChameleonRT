#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include "dx12_utils.h"

// Utilities for DXR ease of use

bool dxr_available(Microsoft::WRL::ComPtr<ID3D12Device5> &device);

// TODO Will: Split this class up some more, this object is more of a "builder"
// while the RootSignature class should really be the "compiled" root signature
// with the final parameter order mapping
class RootSignature {
	D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> sig;

	std::vector<D3D12_ROOT_PARAMETER> params;
	std::vector<D3D12_DESCRIPTOR_RANGE> ranges;

	void add_descriptor(D3D12_ROOT_PARAMETER_TYPE desc_type, uint32_t shader_register, uint32_t space);
	void add_range(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);

public:
	static RootSignature global();
	static RootSignature local();

	void add_constants(uint32_t shader_register, uint32_t space, uint32_t num_vals);

	void add_srv(uint32_t shader_register, uint32_t space);
	void add_uav(uint32_t shader_register, uint32_t space);
	void add_cbv(uint32_t shader_register, uint32_t space);

	void add_srv_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	void add_uav_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	void add_cbv_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	void add_sampler_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);

	void create(ID3D12Device *device);

	ID3D12RootSignature* operator->();
	ID3D12RootSignature* get();
};
