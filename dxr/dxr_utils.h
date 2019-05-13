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
	friend class RTPipelineBuilder;
	friend class RTPipeline;

	RootSignature(D3D12_ROOT_SIGNATURE_FLAGS flags, Microsoft::WRL::ComPtr<ID3D12RootSignature> sig,
		const std::vector<RootParam> &params);

public:
	RootSignature() = default;

	// Returns size_t max if no such param
	size_t offset(const std::string &name) const;
	size_t size(const std::string &name) const;

	size_t descriptor_table_offset() const;
	size_t descriptor_table_size() const;

	// Return the total size of the root signature arguments
	size_t total_size() const;

	ID3D12RootSignature* operator->();
	ID3D12RootSignature* get();
};

// The table doesn't need a name since
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

class ShaderLibrary {
	D3D12_SHADER_BYTECODE bytecode = { 0 };
	D3D12_DXIL_LIBRARY_DESC slibrary = { 0 };

	std::vector<std::wstring> export_functions;
	// A bit annoying but we keep this around too b/c we need a contiguous
	// array of pointers for now to build the exports association in the pipeline
	// TODO: We don't need to keep this
	std::vector<LPCWSTR> export_fcn_ptrs;
	std::vector<D3D12_EXPORT_DESC> exports;

public:
	ShaderLibrary(const void *bytecode, const size_t bytecode_size,
		const std::vector<std::wstring> &exports);
	ShaderLibrary(const ShaderLibrary &other);
	ShaderLibrary& operator=(const ShaderLibrary &other);

	const std::vector<std::wstring>& export_names() const;
	size_t num_exports() const;
	LPCWSTR* export_names_ptr();
	LPCWSTR* find_export(const std::wstring &name);

	const D3D12_DXIL_LIBRARY_DESC* library() const;

private:
	void build_library_desc();
};

struct RootSignatureAssociation {
	std::vector<std::wstring> shaders;
	RootSignature signature;

	RootSignatureAssociation() = default;
	RootSignatureAssociation(const std::vector<std::wstring> &shaders, const RootSignature &signature);
};

struct HitGroup {
	std::wstring name, closest_hit, any_hit, intersection;
	D3D12_HIT_GROUP_TYPE type;

	HitGroup() = default;
	HitGroup(const std::wstring &name, D3D12_HIT_GROUP_TYPE type,
		const std::wstring &closest_hit, const std::wstring &any_hit = L"",
		const std::wstring &intersection = L"");

	bool has_any_hit() const;
	bool has_intersection() const;
};

struct ShaderPayloadConfig {
	std::vector<std::wstring> functions;
	D3D12_RAYTRACING_SHADER_CONFIG desc;

	ShaderPayloadConfig() = default;
	ShaderPayloadConfig(const std::vector<std::wstring> &functions,
		uint32_t max_payload_size, uint32_t max_attrib_size);
};

class RTPipeline;

class RTPipelineBuilder {
	std::vector<ShaderLibrary> shader_libs;
	std::wstring ray_gen;
	std::vector<std::wstring> miss_shaders;
	std::vector<std::vector<HitGroup>> hit_groups;
	std::vector<ShaderPayloadConfig> payload_configs;
	std::vector<RootSignatureAssociation> signature_associations;
	RootSignature global_sig;
	uint32_t recursion_depth = 1;

public:
	RTPipelineBuilder& add_shader_library(const ShaderLibrary &library);

	RTPipelineBuilder& set_ray_gen(const std::wstring &ray_gen);

	// Set the miss shader if you only have one ray type
	RTPipelineBuilder& add_miss_shader(const std::wstring &miss_fn);
	// Set the miss shaders for each ray type
	RTPipelineBuilder& add_miss_shader(const std::vector<std::wstring> &miss_fn);
	
	// Set a single hit-group if there's only one ray type
	RTPipelineBuilder& add_hit_group(const HitGroup &hg);
	// Specify the hit-group for each ray type, and/or each instance
	RTPipelineBuilder& add_hit_group(const std::vector<HitGroup> &hg);

	RTPipelineBuilder& configure_shader_payload(const std::vector<std::wstring> &functions,
		uint32_t max_payload_size, uint32_t max_attrib_size);

	RTPipelineBuilder& set_max_recursion(uint32_t depth);

	RTPipelineBuilder& set_shader_root_sig(const std::vector<std::wstring> &functions, const RootSignature &sig);

	RTPipelineBuilder& set_global_root_sig(const RootSignature &sig);

	RTPipeline create(ID3D12Device5 *device);

private:
	bool has_global_root_sig() const;
	size_t compute_num_subobjects(size_t &num_export_associations, size_t &num_associated_fcns) const;
};

class RTPipeline {
	RootSignature rt_global_sig;
	Microsoft::WRL::ComPtr<ID3D12StateObject> state;
	ID3D12StateObjectProperties *pipeline_props = nullptr;

	std::wstring ray_gen;
	std::vector<std::wstring> miss_shaders;
	std::vector<std::wstring> hit_groups;
	std::vector<RootSignatureAssociation> signature_associations;

	size_t shader_record_size = 0,
		miss_table_offset = 0,
		hit_group_table_offset = 0;
	Buffer shader_table;
	std::unordered_map<std::wstring, size_t> record_offsets;
	uint8_t *sbt_mapping = nullptr;

	friend class RTPipelineBuilder;

	RTPipeline(D3D12_STATE_OBJECT_DESC &desc, RootSignature &global_sig,
		const std::wstring &ray_gen, const std::vector<std::wstring> &miss_shaders,
		const std::vector<std::wstring> &hit_groups,
		const std::vector<RootSignatureAssociation> &signature_associations,
		ID3D12Device5 *device);

public:
	RTPipeline() = default;

	void map_shader_table();
	void unmap_shader_table();

	// Get the pointer in the table to a specific shader record. The table must be mapped
	uint8_t* shader_record(const std::wstring &shader);

	/* Get the local root signature assigned to the shader, if any. Returns null
	 * if no local root signature was set for the shader
	 */
	const RootSignature* shader_signature(const std::wstring &shader) const;

	D3D12_DISPATCH_RAYS_DESC dispatch_rays(const glm::uvec2 &img_dims);

	bool has_global_root_sig() const;
	ID3D12RootSignature* global_sig();

	ID3D12StateObject* operator->();
	ID3D12StateObject* get();

private:
	size_t compute_shader_record_size() const;
};
