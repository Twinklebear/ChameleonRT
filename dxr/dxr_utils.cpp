#include <algorithm>
#include <limits>
#include "util.h"
#include "dxr_utils.h"

using Microsoft::WRL::ComPtr;

bool dxr_available(ComPtr<ID3D12Device5> &device) {
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_data = { 0 };
	CHECK_ERR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
		&feature_data, sizeof(feature_data)));

	return feature_data.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}

RootParam::RootParam(D3D12_ROOT_PARAMETER param, const std::string &name)
	: param(param), name(name)
{}

RootSignature::RootSignature(D3D12_ROOT_SIGNATURE_FLAGS flags, Microsoft::WRL::ComPtr<ID3D12RootSignature> sig,
	const std::vector<RootParam> &params)
	: flags(flags), sig(sig)
{
	size_t offset = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	for (const auto &ip : params) {
		RootParam p = ip;
		p.offset = offset;
		if (p.param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
			// Constants must pad to a size multiple of 8 to align w/ the pointer entries
			p.size = align_to(p.param.Constants.Num32BitValues * 4, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
		} else {
			p.size = sizeof(D3D12_GPU_DESCRIPTOR_HANDLE);
		}
		param_offsets[p.name] = p;
		offset += p.size;

		std::cout << "Param: " << p.name << " is at offset "
			<< p.offset << ", size: " << p.size << "\n";
	}
}

size_t RootSignature::offset(const std::string &name) const {
	auto fnd = param_offsets.find(name);
	if (fnd != param_offsets.end()) {
		return fnd->second.offset;
	} else {
		return std::numeric_limits<size_t>::max();
	}
}

size_t RootSignature::size(const std::string &name) const {
	auto fnd = param_offsets.find(name);
	if (fnd != param_offsets.end()) {
		return fnd->second.size;
	}
	else {
		return std::numeric_limits<size_t>::max();
	}
}

size_t RootSignature::descriptor_table_offset() const {
	return offset("dxr_helper_desc_table");
}

size_t RootSignature::descriptor_table_size() const {
	// We know how big this will be, but it's just for convenience
	return 8;
}


ID3D12RootSignature* RootSignature::operator->() {
	return get();
}

ID3D12RootSignature* RootSignature::get() {
	return sig.Get();
}


RootSignatureBuilder RootSignatureBuilder::global() {
	RootSignatureBuilder sig;
	sig.flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	return sig;
}
RootSignatureBuilder RootSignatureBuilder::local() {
	RootSignatureBuilder sig;
	sig.flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
	return sig;
}

void RootSignatureBuilder::add_descriptor(D3D12_ROOT_PARAMETER_TYPE desc_type, const std::string &name,
	uint32_t shader_register, uint32_t space)
{
	D3D12_ROOT_PARAMETER p = { 0 };
	p.ParameterType = desc_type;
	p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	p.Descriptor.ShaderRegister = shader_register;
	p.Descriptor.RegisterSpace = space;
	params.push_back(RootParam(p, name));
}

void RootSignatureBuilder::add_range(D3D12_DESCRIPTOR_RANGE_TYPE type,
	uint32_t size, uint32_t base_register, uint32_t space, uint32_t table_offset)
{
	D3D12_DESCRIPTOR_RANGE r = { 0 };
	r.RangeType = type;
	r.NumDescriptors = size;
	r.BaseShaderRegister = base_register;
	r.RegisterSpace = space;
	r.OffsetInDescriptorsFromTableStart = table_offset;
	ranges.push_back(r);
}

RootSignatureBuilder& RootSignatureBuilder::add_constants(const std::string &name, uint32_t shader_register,
	uint32_t space, uint32_t num_vals)
{
	D3D12_ROOT_PARAMETER p = { 0 };
	p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	p.Constants.ShaderRegister = shader_register;
	p.Constants.RegisterSpace = space;
	p.Constants.Num32BitValues = num_vals;
	params.push_back(RootParam(p, name));
	return *this;
}

RootSignatureBuilder& RootSignatureBuilder::add_srv(const std::string &name, uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, name, shader_register, space);
	return *this;
}
RootSignatureBuilder& RootSignatureBuilder::add_uav(const std::string &name, uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV, name, shader_register, space);
	return *this;
}
RootSignatureBuilder& RootSignatureBuilder::add_cbv(const std::string &name, uint32_t shader_register, uint32_t space) {
	add_descriptor(D3D12_ROOT_PARAMETER_TYPE_CBV, name, shader_register, space);
	return *this;
}

RootSignatureBuilder& RootSignatureBuilder::add_srv_range(uint32_t size, uint32_t base_register,
	uint32_t space, uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, size, base_register, space, table_offset);
	return *this;
}
RootSignatureBuilder& RootSignatureBuilder::add_uav_range(uint32_t size, uint32_t base_register,
	uint32_t space, uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, size, base_register, space, table_offset);
	return *this;
}
RootSignatureBuilder& RootSignatureBuilder::add_cbv_range(uint32_t size, uint32_t base_register,
	uint32_t space, uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, size, base_register, space, table_offset);
	return *this;
}
RootSignatureBuilder& RootSignatureBuilder::add_sampler_range(uint32_t size, uint32_t base_register,
	uint32_t space, uint32_t table_offset)
{
	add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, size, base_register, space, table_offset);
	return *this;
}

RootSignature RootSignatureBuilder::create(ID3D12Device *device) {
	// Build the set of root parameters from the inputs
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
	std::stable_partition(params.begin(), params.end(),
		[](const RootParam &p) {
			return p.param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	});

	if (!ranges.empty()) {
		// Append table the descriptor table parameter
		D3D12_ROOT_PARAMETER desc_table = { 0 };
		desc_table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		desc_table.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		desc_table.DescriptorTable.NumDescriptorRanges = ranges.size();
		desc_table.DescriptorTable.pDescriptorRanges = ranges.data();
		params.push_back(RootParam(desc_table, "dxr_helper_desc_table"));
	}

	std::vector<D3D12_ROOT_PARAMETER> all_params;
	std::transform(params.begin(), params.end(), std::back_inserter(all_params),
		[](const RootParam &p) { return p.param; });

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

	ComPtr<ID3D12RootSignature> signature;
	CHECK_ERR(device->CreateRootSignature(0, signature_blob->GetBufferPointer(),
		signature_blob->GetBufferSize(), IID_PPV_ARGS(&signature)));

	return RootSignature(flags, signature, params);
}

ShaderLibrary::ShaderLibrary(const void *code, const size_t code_size,
	const std::vector<std::wstring> &export_fns)
	: export_functions(export_fns)
{
	bytecode.pShaderBytecode = code;
	bytecode.BytecodeLength = code_size;
	build_library_desc();
}

ShaderLibrary::ShaderLibrary(const ShaderLibrary &other)
	: bytecode(other.bytecode), export_functions(other.export_functions)
{
	build_library_desc();
}

ShaderLibrary& ShaderLibrary::operator=(const ShaderLibrary &other) {
	bytecode = other.bytecode;
	export_functions = other.export_functions;
	build_library_desc();
	return *this;
}

size_t ShaderLibrary::num_exports() const {
	return export_fcn_ptrs.size();
}

LPCWSTR* ShaderLibrary::export_names() {
	return export_fcn_ptrs.data();
}

LPCWSTR* ShaderLibrary::find_export(const std::wstring &name) {
	auto fnd = std::find(export_functions.begin(), export_functions.end(), name);
	if (fnd != export_functions.end()) {
		size_t idx = std::distance(export_functions.begin(), fnd);
		return &export_fcn_ptrs[idx];
	} else {
		return nullptr;
	}
}

const D3D12_DXIL_LIBRARY_DESC* ShaderLibrary::library() const {
	return &slibrary;
}

void ShaderLibrary::build_library_desc() {
	for (const auto &fn : export_functions) {
		D3D12_EXPORT_DESC shader_export = { 0 };
		shader_export.ExportToRename = nullptr;
		shader_export.Flags = D3D12_EXPORT_FLAG_NONE;
		shader_export.Name = fn.c_str();
		exports.push_back(shader_export);
		export_fcn_ptrs.push_back(fn.c_str());
	}
	slibrary.DXILLibrary = bytecode;
	slibrary.NumExports = exports.size();
	slibrary.pExports = exports.data();
}
