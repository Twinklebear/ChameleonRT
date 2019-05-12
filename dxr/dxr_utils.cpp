#include <algorithm>
#include "dxr_utils.h"

using Microsoft::WRL::ComPtr;

bool dxr_available(ComPtr<ID3D12Device5> &device) {
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_data = { 0 };
	CHECK_ERR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
		&feature_data, sizeof(feature_data)));

	return feature_data.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}

RootSignature RootSignature::global() {
	RootSignature sig;
	sig.flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	return sig;
}
RootSignature RootSignature::local() {
	RootSignature sig;
	sig.flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
	return sig;
}

void RootSignature::add_descriptor(D3D12_ROOT_PARAMETER_TYPE desc_type, uint32_t shader_register, uint32_t space) {
	D3D12_ROOT_PARAMETER p = { 0 };
	p.ParameterType = desc_type;
	p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	p.Descriptor.ShaderRegister = shader_register;
	p.Descriptor.RegisterSpace = space;
	params.push_back(p);
}

void RootSignature::add_range(D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t size, uint32_t base_register, uint32_t space,
	uint32_t table_offset)
{
	D3D12_DESCRIPTOR_RANGE r = { 0 };
	r.RangeType = type;
	r.NumDescriptors = size;
	r.BaseShaderRegister = base_register;
	r.RegisterSpace = space;
	r.OffsetInDescriptorsFromTableStart = table_offset;
	ranges.push_back(r);
}

void RootSignature::add_constants(uint32_t shader_register, uint32_t space, uint32_t num_vals) {
	D3D12_ROOT_PARAMETER p = { 0 };
	p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	p.Constants.ShaderRegister = shader_register;
	p.Constants.RegisterSpace = space;
	p.Constants.Num32BitValues = num_vals;
	params.push_back(p);
}

void RootSignature::add_srv(uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, shader_register, space);
}
void RootSignature::add_uav(uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV, shader_register, space);
}
void RootSignature::add_cbv(uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_CBV, shader_register, space);
}

void RootSignature::add_srv_range(uint32_t size, uint32_t base_register, uint32_t space,
	uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, size, base_register, space, table_offset);
}
void RootSignature::add_uav_range(uint32_t size, uint32_t base_register, uint32_t space,
	uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, size, base_register, space, table_offset);
}
void RootSignature::add_cbv_range(uint32_t size, uint32_t base_register, uint32_t space,
	uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, size, base_register, space, table_offset);
}
void RootSignature::add_sampler_range(uint32_t size, uint32_t base_register, uint32_t space,
	uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, size, base_register, space, table_offset);
}

void RootSignature::create(ID3D12Device *device) {
	// Build the set of root parameters from the inputs
	std::vector<D3D12_ROOT_PARAMETER> all_params = params;
	// Pack constant values to the front, since we want to compact the shader record
	// to avoid a layout where we have something like the following:
	// [constant, pad]
	// [pointer]
	// [constant, pad]
	// since we could instead have done:
	// [constant, constant]
	// [pointer]
	// TODO WILL: Now I do need a name to associate with these params, since after I re-shuffle
	// them all around the order may not the one the add* calls were made, so the shader
	// record needs this info to setup the params properly
	std::stable_partition(all_params.begin(), all_params.end(),
		[](const D3D12_ROOT_PARAMETER &p) {
			return p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	});

	if (!ranges.empty()) {
		// Append table the descriptor table parameter
		D3D12_ROOT_PARAMETER desc_table = { 0 };
		desc_table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		desc_table.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		desc_table.DescriptorTable.NumDescriptorRanges = ranges.size();
		desc_table.DescriptorTable.pDescriptorRanges = ranges.data();
		all_params.push_back(desc_table);
	}

	D3D12_ROOT_SIGNATURE_DESC root_desc = { 0 };
	root_desc.NumParameters = all_params.size();
	root_desc.pParameters = all_params.data();
	root_desc.Flags = flags;

	// Create the root signature from the descriptor
	ComPtr<ID3DBlob> signature_blob;
	ComPtr<ID3DBlob> err_blob;
	auto res = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
		&signature_blob, &err_blob);
	if (FAILED(res)) {
		std::cout << "Failed to serialize root signature: " << err_blob->GetBufferPointer() << "\n";
		throw std::runtime_error("Failed to serialize root signature");
	}

	CHECK_ERR(device->CreateRootSignature(0, signature_blob->GetBufferPointer(),
		signature_blob->GetBufferSize(), IID_PPV_ARGS(&sig)));
}

ID3D12RootSignature* RootSignature::operator->() {
	return get();
}

ID3D12RootSignature* RootSignature::get() {
	return sig.Get();
}
