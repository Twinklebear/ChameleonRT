#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include "dx12_utils.h"

// Utilities for DXR ease of use

bool dxr_available(Microsoft::WRL::ComPtr<ID3D12Device5> &device);

class RootSignatureBuilder;

struct RootParam {
	D3D12_ROOT_PARAMETER param = { 0 };
	std::string name;
	size_t offset = 0;
	size_t size = 0;

	RootParam() = default;
	RootParam(D3D12_ROOT_PARAMETER param, const std::string &name);
};

struct RootParamHash;

class RootSignature {
	D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> sig;

	// The offsets of the parameters into the shader arguments part
	// of the shader record. The offsets returned account for the shader identifier size
	std::unordered_map<std::string, RootParam> param_offsets;

	friend class RootSignatureBuilder;

	RootSignature(D3D12_ROOT_SIGNATURE_FLAGS flags, Microsoft::WRL::ComPtr<ID3D12RootSignature> sig,
		const std::vector<RootParam> &params);

public:
	RootSignature() = default;

	// Returns size_t max if no such param
	size_t offset(const std::string &name) const;
	size_t size(const std::string &name) const;

	size_t descriptor_table_offset() const;
	size_t descriptor_table_size() const;

	ID3D12RootSignature* operator->();
	ID3D12RootSignature* get();
};


// TODO Will: Split this class up some more, this object is more of a "builder"
// while the RootSignature class should really be the "compiled" root signature
// with the final parameter order mapping. The table doesn't need a name since
// I think there should only ever be one, since you can specify a bunch of ranges
// to reference the single pointer you get, right?
// Actually since it refers into the global descriptor heap we don't need to write
// anything into the actual shader table. I wonder why the RT gems book and other
// resources seem to imply this? Because it seems like you can't read from a different
// heap than the one which is bound.
class RootSignatureBuilder {
	D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	std::vector<RootParam> params;
	std::vector<D3D12_DESCRIPTOR_RANGE> ranges;

	void add_descriptor(D3D12_ROOT_PARAMETER_TYPE desc_type, const std::string &name,
		uint32_t shader_register, uint32_t space);
	void add_range(D3D12_DESCRIPTOR_RANGE_TYPE type,
		uint32_t size, uint32_t base_register, uint32_t space, uint32_t table_offset);

public:
	static RootSignatureBuilder global();
	static RootSignatureBuilder local();

	RootSignatureBuilder& add_constants(const std::string &name, uint32_t shader_register,
		uint32_t space, uint32_t num_vals);

	RootSignatureBuilder& add_srv(const std::string &name, uint32_t shader_register, uint32_t space);
	RootSignatureBuilder& add_uav(const std::string &name, uint32_t shader_register, uint32_t space);
	RootSignatureBuilder& add_cbv(const std::string &name, uint32_t shader_register, uint32_t space);

	RootSignatureBuilder& add_srv_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	RootSignatureBuilder& add_uav_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	RootSignatureBuilder& add_cbv_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);
	RootSignatureBuilder& add_sampler_range(uint32_t size, uint32_t base_register, uint32_t space,
		uint32_t table_offset);

	RootSignature create(ID3D12Device *device);
};
