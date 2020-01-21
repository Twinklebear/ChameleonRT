#include "dxr_utils.h"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include "mesh.h"
#include "util.h"

namespace dxr {

using Microsoft::WRL::ComPtr;

bool dxr_available(ComPtr<ID3D12Device5> &device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_data = {0};
    CHECK_ERR(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS5, &feature_data, sizeof(feature_data)));

    return feature_data.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}

RootParam::RootParam(D3D12_ROOT_PARAMETER param, const std::string &name)
    : param(param), name(name)
{
}

DescriptorHeap::DescriptorHeap(D3D12_DESCRIPTOR_HEAP_DESC desc,
                               std::vector<D3D12_DESCRIPTOR_RANGE> ranges,
                               Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap)
    : desc(desc), ranges(ranges), heap(heap)
{
}

D3D12_ROOT_PARAMETER DescriptorHeap::root_param() const
{
    D3D12_ROOT_PARAMETER param = {0};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    param.DescriptorTable.NumDescriptorRanges = ranges.size();
    param.DescriptorTable.pDescriptorRanges = ranges.data();
    return param;
}

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::gpu_desc_handle()
{
    return heap->GetGPUDescriptorHandleForHeapStart();
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::cpu_desc_handle()
{
    return heap->GetCPUDescriptorHandleForHeapStart();
}

ID3D12DescriptorHeap *DescriptorHeap::operator->()
{
    return heap.Get();
}

ID3D12DescriptorHeap *DescriptorHeap::get()
{
    return heap.Get();
}

void DescriptorHeapBuilder::add_range(D3D12_DESCRIPTOR_RANGE_TYPE type,
                                      uint32_t size,
                                      uint32_t base_register,
                                      uint32_t space)
{
    D3D12_DESCRIPTOR_RANGE r = {0};
    r.RangeType = type;
    r.NumDescriptors = size;
    r.BaseShaderRegister = base_register;
    r.RegisterSpace = space;
    r.OffsetInDescriptorsFromTableStart = num_descriptors();
    ranges.push_back(r);
}

bool DescriptorHeapBuilder::contains_range_type(D3D12_DESCRIPTOR_RANGE_TYPE type)
{
    return std::find_if(ranges.begin(), ranges.end(), [&](const D3D12_DESCRIPTOR_RANGE &r) {
               return r.RangeType == type;
           }) != ranges.end();
}

uint32_t DescriptorHeapBuilder::num_descriptors()
{
    return std::accumulate(ranges.begin(),
                           ranges.end(),
                           0,
                           [](const uint32_t &n, const D3D12_DESCRIPTOR_RANGE &r) {
                               return n + r.NumDescriptors;
                           });
}

DescriptorHeapBuilder &DescriptorHeapBuilder::add_srv_range(uint32_t size,
                                                            uint32_t base_register,
                                                            uint32_t space)
{
    add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, size, base_register, space);
    return *this;
}
DescriptorHeapBuilder &DescriptorHeapBuilder::add_uav_range(uint32_t size,
                                                            uint32_t base_register,
                                                            uint32_t space)
{
    add_range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, size, base_register, space);
    return *this;
}
DescriptorHeapBuilder &DescriptorHeapBuilder::add_cbv_range(uint32_t size,
                                                            uint32_t base_register,
                                                            uint32_t space)
{
    add_range(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, size, base_register, space);
    return *this;
}
DescriptorHeapBuilder &DescriptorHeapBuilder::add_sampler_range(uint32_t size,
                                                                uint32_t base_register,
                                                                uint32_t space)
{
    add_range(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, size, base_register, space);
    return *this;
}

DescriptorHeap DescriptorHeapBuilder::create(ID3D12Device *device)
{
    const bool contains_cbv_srv_uav = contains_range_type(D3D12_DESCRIPTOR_RANGE_TYPE_CBV) ||
                                      contains_range_type(D3D12_DESCRIPTOR_RANGE_TYPE_SRV) ||
                                      contains_range_type(D3D12_DESCRIPTOR_RANGE_TYPE_UAV);
    if (contains_cbv_srv_uav && contains_range_type(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)) {
        throw std::runtime_error(
            "Error: CBV/SRV/UAV descriptors cannot be in the same heap as samplers");
    }

    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {0};
    heap_desc.NumDescriptors = num_descriptors();
    heap_desc.Type = contains_cbv_srv_uav ? D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                          : D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_ERR(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)));
    return DescriptorHeap(heap_desc, ranges, heap);
}

RootSignature::RootSignature(D3D12_ROOT_SIGNATURE_FLAGS flags,
                             Microsoft::WRL::ComPtr<ID3D12RootSignature> sig,
                             const std::vector<RootParam> &params)
    : flags(flags), sig(sig)
{
    size_t offset = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    for (const auto &ip : params) {
        RootParam p = ip;
        p.offset = offset;
        if (p.param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
            // Constants must pad to a size multiple of 8 to align w/ the pointer entries
            p.size = align_to(p.param.Constants.Num32BitValues * 4,
                              sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
        } else {
            p.size = sizeof(D3D12_GPU_DESCRIPTOR_HANDLE);
        }
        param_offsets[p.name] = p;
        offset += p.size;
    }
}

size_t RootSignature::offset(const std::string &name) const
{
    auto fnd = param_offsets.find(name);
    if (fnd != param_offsets.end()) {
        return fnd->second.offset;
    } else {
        throw std::runtime_error("Request for offset to " + name +
                                 " which is not in the root signature");
        return std::numeric_limits<size_t>::max();
    }
}

size_t RootSignature::size(const std::string &name) const
{
    auto fnd = param_offsets.find(name);
    if (fnd != param_offsets.end()) {
        return fnd->second.size;
    } else {
        return std::numeric_limits<size_t>::max();
    }
}

size_t RootSignature::total_size() const
{
    return std::accumulate(param_offsets.begin(),
                           param_offsets.end(),
                           size_t(0),
                           [](const size_t &n, const auto &p) { return n + p.second.size; });
}

ID3D12RootSignature *RootSignature::operator->()
{
    return get();
}

ID3D12RootSignature *RootSignature::get()
{
    return sig.Get();
}

RootSignatureBuilder RootSignatureBuilder::global()
{
    RootSignatureBuilder sig;
    sig.flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    return sig;
}
RootSignatureBuilder RootSignatureBuilder::local()
{
    RootSignatureBuilder sig;
    sig.flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
    return sig;
}

void RootSignatureBuilder::add_descriptor(D3D12_ROOT_PARAMETER_TYPE desc_type,
                                          const std::string &name,
                                          uint32_t shader_register,
                                          uint32_t space)
{
    D3D12_ROOT_PARAMETER p = {0};
    p.ParameterType = desc_type;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p.Descriptor.ShaderRegister = shader_register;
    p.Descriptor.RegisterSpace = space;
    params.push_back(RootParam(p, name));
}

RootSignatureBuilder &RootSignatureBuilder::add_constants(const std::string &name,
                                                          uint32_t shader_register,
                                                          uint32_t num_vals,
                                                          uint32_t space)
{
    D3D12_ROOT_PARAMETER p = {0};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p.Constants.ShaderRegister = shader_register;
    p.Constants.RegisterSpace = space;
    p.Constants.Num32BitValues = num_vals;
    params.push_back(RootParam(p, name));
    return *this;
}

RootSignatureBuilder &RootSignatureBuilder::add_srv(const std::string &name,
                                                    uint32_t shader_register,
                                                    uint32_t space)
{
    add_descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, name, shader_register, space);
    return *this;
}
RootSignatureBuilder &RootSignatureBuilder::add_uav(const std::string &name,
                                                    uint32_t shader_register,
                                                    uint32_t space)
{
    add_descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV, name, shader_register, space);
    return *this;
}
RootSignatureBuilder &RootSignatureBuilder::add_cbv(const std::string &name,
                                                    uint32_t shader_register,
                                                    uint32_t space)
{
    add_descriptor(D3D12_ROOT_PARAMETER_TYPE_CBV, name, shader_register, space);
    return *this;
}
RootSignatureBuilder &RootSignatureBuilder::add_desc_heap(const std::string &name,
                                                          const DescriptorHeap &heap)
{
    params.push_back(RootParam(heap.root_param(), name));
    return *this;
}

RootSignature RootSignatureBuilder::create(ID3D12Device *device)
{
    // Build the set of root parameters from the inputs
    // Pack constant values to the front, since we want to compact the shader record
    // to avoid a layout where we have something like the following:
    // [constant, pad]
    // [pointer]
    // [constant, pad]
    // since we could instead have done:
    // [constant, constant]
    // [pointer]
    std::stable_partition(params.begin(), params.end(), [](const RootParam &p) {
        return p.param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    });

    std::vector<D3D12_ROOT_PARAMETER> all_params;
    std::transform(
        params.begin(), params.end(), std::back_inserter(all_params), [](const RootParam &p) {
            return p.param;
        });

    D3D12_ROOT_SIGNATURE_DESC root_desc = {0};
    root_desc.NumParameters = all_params.size();
    root_desc.pParameters = all_params.data();
    root_desc.Flags = flags;

    // Create the root signature from the descriptor
    ComPtr<ID3DBlob> signature_blob;
    ComPtr<ID3DBlob> err_blob;
    auto res = D3D12SerializeRootSignature(
        &root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature_blob, &err_blob);
    if (FAILED(res)) {
        std::cout << "Failed to serialize root signature: "
                  << static_cast<char *>(err_blob->GetBufferPointer()) << "\n";
        throw std::runtime_error("Failed to serialize root signature");
    }

    ComPtr<ID3D12RootSignature> signature;
    CHECK_ERR(device->CreateRootSignature(0,
                                          signature_blob->GetBufferPointer(),
                                          signature_blob->GetBufferSize(),
                                          IID_PPV_ARGS(&signature)));

    return RootSignature(flags, signature, params);
}

ShaderLibrary::ShaderLibrary(const void *code,
                             const size_t code_size,
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

ShaderLibrary &ShaderLibrary::operator=(const ShaderLibrary &other)
{
    bytecode = other.bytecode;
    export_functions = other.export_functions;
    build_library_desc();
    return *this;
}

const std::vector<std::wstring> &ShaderLibrary::export_names() const
{
    return export_functions;
}

size_t ShaderLibrary::num_exports() const
{
    return export_fcn_ptrs.size();
}

LPCWSTR *ShaderLibrary::export_names_ptr()
{
    return export_fcn_ptrs.data();
}

LPCWSTR *ShaderLibrary::find_export(const std::wstring &name)
{
    auto fnd = std::find(export_functions.begin(), export_functions.end(), name);
    if (fnd != export_functions.end()) {
        size_t idx = std::distance(export_functions.begin(), fnd);
        return &export_fcn_ptrs[idx];
    } else {
        return nullptr;
    }
}

const D3D12_DXIL_LIBRARY_DESC *ShaderLibrary::library() const
{
    return &slibrary;
}

void ShaderLibrary::build_library_desc()
{
    for (const auto &fn : export_functions) {
        D3D12_EXPORT_DESC shader_export = {0};
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

RootSignatureAssociation::RootSignatureAssociation(const std::vector<std::wstring> &shaders,
                                                   const RootSignature &signature)
    : shaders(shaders), signature(signature)
{
}

HitGroup::HitGroup(const std::wstring &name,
                   D3D12_HIT_GROUP_TYPE type,
                   const std::wstring &closest_hit,
                   const std::wstring &any_hit,
                   const std::wstring &intersection)
    : name(name),
      closest_hit(closest_hit),
      type(type),
      any_hit(any_hit),
      intersection(intersection)
{
}

bool HitGroup::has_any_hit() const
{
    return !any_hit.empty();
}
bool HitGroup::has_intersection() const
{
    return !intersection.empty();
}

ShaderPayloadConfig::ShaderPayloadConfig(const std::vector<std::wstring> &functions,
                                         uint32_t max_payload_size,
                                         uint32_t max_attrib_size)
    : functions(functions)
{
    desc.MaxPayloadSizeInBytes = max_payload_size;
    desc.MaxAttributeSizeInBytes = max_attrib_size;
}

RTPipelineBuilder &RTPipelineBuilder::add_shader_library(const ShaderLibrary &library)
{
    shader_libs.push_back(library);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_ray_gen(const std::wstring &rg)
{
    assert(ray_gen.empty());
    ray_gen = rg;
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_miss_shader(const std::wstring &miss_fn)
{
    miss_shaders.push_back(miss_fn);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_hit_group(const HitGroup &hg)
{
    hit_groups.push_back({hg});
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::configure_shader_payload(
    const std::vector<std::wstring> &functions,
    uint32_t max_payload_size,
    uint32_t max_attrib_size)
{
    payload_configs.emplace_back(functions, max_payload_size, max_attrib_size);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_max_recursion(uint32_t depth)
{
    recursion_depth = depth;
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_shader_root_sig(
    const std::vector<std::wstring> &functions, const RootSignature &sig)
{
    signature_associations.emplace_back(functions, sig);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_global_root_sig(const RootSignature &sig)
{
    global_sig = sig;
    return *this;
}

RTPipeline RTPipelineBuilder::create(ID3D12Device5 *device)
{
    if (ray_gen.empty()) {
        throw std::runtime_error("No ray generation shader set!");
    }

    size_t num_association_subobjs = 0;
    size_t num_associated_fcns = 0;
    const size_t total_subobjs =
        compute_num_subobjects(num_association_subobjs, num_associated_fcns);

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    subobjects.resize(total_subobjs);
    size_t current_obj = 0;

    std::vector<D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION> associations;
    associations.resize(num_association_subobjs);
    size_t current_assoc = 0;

    std::vector<LPCWSTR> associated_fcns;
    associated_fcns.resize(num_associated_fcns, nullptr);
    size_t current_assoc_fcn = 0;

    // Add the shader libraries
    for (const auto &lib : shader_libs) {
        D3D12_STATE_SUBOBJECT l = {0};
        l.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        l.pDesc = lib.library();
        subobjects[current_obj++] = l;
    }

    // Make the hit group descriptors for each hit group and ray type and add them
    std::vector<D3D12_HIT_GROUP_DESC> hg_descs;
    // Names for the RTPipeline to setup the shader table with
    std::vector<std::wstring> hit_group_names;
    if (!hit_groups.empty()) {
        hg_descs.resize(hit_groups.size());
        for (size_t i = 0; i < hit_groups.size(); ++i) {
            const auto &hg = hit_groups[i];

            hit_group_names.push_back(hg.name);

            D3D12_HIT_GROUP_DESC &desc = hg_descs[i];
            desc.HitGroupExport = hg.name.c_str();
            desc.Type = hg.type;
            desc.ClosestHitShaderImport = hg.closest_hit.c_str();
            desc.IntersectionShaderImport =
                hg.has_intersection() ? hg.intersection.c_str() : nullptr;
            desc.AnyHitShaderImport = hg.has_any_hit() ? hg.any_hit.c_str() : nullptr;

            D3D12_STATE_SUBOBJECT o = {0};
            o.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
            o.pDesc = &desc;
            subobjects[current_obj++] = o;
        }
    }

    // Make the shader payload configs and associate them with the desired functions
    for (const auto &c : payload_configs) {
        // Add the shader config object
        D3D12_STATE_SUBOBJECT o = {0};
        o.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
        o.pDesc = &c.desc;
        subobjects[current_obj++] = o;

        // Associate it with the exports
        D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION &assoc = associations[current_assoc++];
        assoc.NumExports = c.functions.size();
        assoc.pExports = &associated_fcns[current_assoc_fcn];
        assoc.pSubobjectToAssociate = &subobjects[current_obj - 1];

        // Copy over the names referenced by this association
        for (const auto &name : c.functions) {
            associated_fcns[current_assoc_fcn++] = name.c_str();
        }

        D3D12_STATE_SUBOBJECT payload_subobj = {0};
        payload_subobj.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        payload_subobj.pDesc = &assoc;
        subobjects[current_obj++] = payload_subobj;
    }

    // Make the local root signature objects and associations
    std::vector<D3D12_LOCAL_ROOT_SIGNATURE> local_root_sigs;
    if (!signature_associations.empty()) {
        local_root_sigs.resize(signature_associations.size());
        size_t current_sig = 0;
        for (auto &sig : signature_associations) {
            // Add the local root signature
            D3D12_LOCAL_ROOT_SIGNATURE &local_sig = local_root_sigs[current_sig++];
            local_sig.pLocalRootSignature = sig.signature.get();

            D3D12_STATE_SUBOBJECT root_sig_obj = {0};
            root_sig_obj.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
            root_sig_obj.pDesc = &local_sig;
            subobjects[current_obj++] = root_sig_obj;

            // Associate it with the exports
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION &assoc = associations[current_assoc++];
            assoc.NumExports = sig.shaders.size();
            assoc.pExports = &associated_fcns[current_assoc_fcn];
            assoc.pSubobjectToAssociate = &subobjects[current_obj - 1];

            // Copy over the names referenced by this association
            for (const auto &name : sig.shaders) {
                associated_fcns[current_assoc_fcn++] = name.c_str();
            }

            D3D12_STATE_SUBOBJECT payload_subobj = {0};
            payload_subobj.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
            payload_subobj.pDesc = &assoc;
            subobjects[current_obj++] = payload_subobj;
        }
    }

    // Add the raytracing pipeline config
    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_cfg = {0};
    pipeline_cfg.MaxTraceRecursionDepth = 1;
    {
        D3D12_STATE_SUBOBJECT o = {0};
        o.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
        o.pDesc = &pipeline_cfg;
        // 8: Pipeline config
        subobjects[current_obj++] = o;
    }

    // Add the global root signature if we have one
    D3D12_GLOBAL_ROOT_SIGNATURE global_root_sig_obj = {0};
    if (has_global_root_sig()) {
        global_root_sig_obj.pGlobalRootSignature = global_sig.get();
        D3D12_STATE_SUBOBJECT o = {0};
        o.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        o.pDesc = &global_root_sig_obj;
        subobjects[current_obj++] = o;
    }

    D3D12_STATE_OBJECT_DESC pipeline_desc = {0};
    pipeline_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    pipeline_desc.NumSubobjects = current_obj;
    pipeline_desc.pSubobjects = subobjects.data();

    return RTPipeline(pipeline_desc,
                      global_sig,
                      ray_gen,
                      miss_shaders,
                      hit_group_names,
                      signature_associations,
                      device);
}

bool RTPipelineBuilder::has_global_root_sig() const
{
    return global_sig.sig.Get() != nullptr;
}

size_t RTPipelineBuilder::compute_num_subobjects(size_t &num_export_associations,
                                                 size_t &num_associated_fcns) const
{
    // Compute how many state objects we'll need for this pipeline
    // Each DXIL library takes one subobject
    size_t num_subobjs = shader_libs.size();

    // Each hit group takes one subobject
    num_subobjs += hit_groups.size();

    // Each shader payload config takes two subobjects:
    // One to declare the config, and another to associate it with the functions
    num_subobjs += payload_configs.size() * 2;
    num_export_associations = payload_configs.size();
    num_associated_fcns = 0;
    for (const auto &c : payload_configs) {
        num_associated_fcns += c.functions.size();
    }

    // Each local root signature association takes two subobjects:
    // One to declare the subobject, and another to associate it with the functions
    num_subobjs += signature_associations.size() * 2;
    num_export_associations += signature_associations.size();
    for (const auto &a : signature_associations) {
        num_associated_fcns += a.shaders.size();
    }

    // Specifying the max trace recursion depth takes 1 subobject
    ++num_subobjs;

    // If we have a global root signature that takes 1 subobject
    if (has_global_root_sig()) {
        ++num_subobjs;
    }
    return num_subobjs;
}

RTPipeline::RTPipeline(D3D12_STATE_OBJECT_DESC &desc,
                       RootSignature &global_sig,
                       const std::wstring &ray_gen,
                       const std::vector<std::wstring> &miss_shaders,
                       const std::vector<std::wstring> &hit_groups,
                       const std::vector<RootSignatureAssociation> &signature_associations,
                       ID3D12Device5 *device)
    : rt_global_sig(global_sig),
      ray_gen(ray_gen),
      miss_shaders(miss_shaders),
      hit_groups(hit_groups),
      signature_associations(signature_associations)
{
    CHECK_ERR(device->CreateStateObject(&desc, IID_PPV_ARGS(&state)));
    CHECK_ERR(state->QueryInterface(&pipeline_props));

    // Compute the offsets/strides for each set of shaders in the SBT
    const size_t raygen_record_offset = 0;
    dispatch_desc.RayGenerationShaderRecord.SizeInBytes = compute_shader_record_size(ray_gen);

    const size_t miss_record_offset =
        align_to(dispatch_desc.RayGenerationShaderRecord.SizeInBytes,
                 D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatch_desc.MissShaderTable.StrideInBytes = 0;
    for (const auto &m : miss_shaders) {
        dispatch_desc.MissShaderTable.StrideInBytes = std::max(
            dispatch_desc.MissShaderTable.StrideInBytes, compute_shader_record_size(m));
    }
    dispatch_desc.MissShaderTable.SizeInBytes =
        dispatch_desc.MissShaderTable.StrideInBytes * miss_shaders.size();

    const size_t hitgroup_record_offset =
        align_to(miss_record_offset + dispatch_desc.MissShaderTable.SizeInBytes,
                 D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    dispatch_desc.HitGroupTable.StrideInBytes = 0;
    for (const auto &hg : hit_groups) {
        dispatch_desc.HitGroupTable.StrideInBytes = std::max(
            dispatch_desc.HitGroupTable.StrideInBytes, compute_shader_record_size(hg));
    }
    dispatch_desc.HitGroupTable.SizeInBytes =
        dispatch_desc.HitGroupTable.StrideInBytes * hit_groups.size();

    const size_t sbt_size = hitgroup_record_offset + dispatch_desc.HitGroupTable.SizeInBytes;

    cpu_shader_table = Buffer::upload(device, sbt_size, D3D12_RESOURCE_STATE_GENERIC_READ);
    shader_table = Buffer::default(device, sbt_size, D3D12_RESOURCE_STATE_GENERIC_READ);

    // Build the list of offsets into the shader table for each shader record
    // and write the identifiers into the table. The actual arguments are left to the user
    map_shader_table();

    // Write the ray gen shader
    dispatch_desc.RayGenerationShaderRecord.StartAddress =
        shader_table->GetGPUVirtualAddress();
    record_offsets[ray_gen] = 0;
    std::memcpy(sbt_mapping,
                pipeline_props->GetShaderIdentifier(ray_gen.c_str()),
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

    // Write the miss shaders
    dispatch_desc.MissShaderTable.StartAddress =
        shader_table->GetGPUVirtualAddress() + miss_record_offset;
    size_t offset = miss_record_offset;
    for (const auto &m : miss_shaders) {
        record_offsets[m] = offset;
        std::memcpy(sbt_mapping + offset,
                    pipeline_props->GetShaderIdentifier(m.c_str()),
                    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

        offset += dispatch_desc.MissShaderTable.StrideInBytes;
    }

    // Write the hit group shaders
    dispatch_desc.HitGroupTable.StartAddress =
        shader_table->GetGPUVirtualAddress() + hitgroup_record_offset;
    offset = hitgroup_record_offset;
    for (const auto &hg : hit_groups) {
        record_offsets[hg] = offset;
        std::memcpy(sbt_mapping + offset,
                    pipeline_props->GetShaderIdentifier(hg.c_str()),
                    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

        offset += dispatch_desc.HitGroupTable.StrideInBytes;
    }
    unmap_shader_table();
}

void RTPipeline::map_shader_table()
{
    assert(sbt_mapping == nullptr);
    sbt_mapping = static_cast<uint8_t *>(cpu_shader_table.map());
}

void RTPipeline::unmap_shader_table()
{
    assert(sbt_mapping);
    cpu_shader_table.unmap();
    sbt_mapping = nullptr;
}

void RTPipeline::upload_shader_table(ID3D12GraphicsCommandList4 *cmd_list)
{
    assert(sbt_mapping == nullptr);

    auto b = barrier_transition(shader_table, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd_list->ResourceBarrier(1, &b);

    cmd_list->CopyResource(shader_table.get(), cpu_shader_table.get());

    b = barrier_transition(shader_table, D3D12_RESOURCE_STATE_GENERIC_READ);
    cmd_list->ResourceBarrier(1, &b);
}

uint8_t *RTPipeline::shader_record(const std::wstring &shader)
{
    assert(sbt_mapping);
    auto fnd = record_offsets.find(shader);
    if (fnd != record_offsets.end()) {
        return sbt_mapping + fnd->second;
    } else {
        throw std::runtime_error("Request for shader record not in table!");
    }
}

const RootSignature *RTPipeline::shader_signature(const std::wstring &shader) const
{
    // Note: The numbers of shaders and root signatures should be relatively small,
    // but this is O(n^2). For a massive scene update this to use a faster unordered map
    // (like the sparsepp one). However, for a big scene the number of signatures shouldn't be
    // too high still right? Just the number of shaders they're associated with
    auto fnd = std::find_if(signature_associations.begin(),
                            signature_associations.end(),
                            [&](const RootSignatureAssociation &s) {
                                return std::find(s.shaders.begin(), s.shaders.end(), shader) !=
                                       s.shaders.end();
                            });
    if (fnd != signature_associations.end()) {
        return &fnd->signature;
    } else {
        return nullptr;
    }
}

D3D12_DISPATCH_RAYS_DESC RTPipeline::dispatch_rays(const glm::uvec2 &img_dims)
{
    D3D12_DISPATCH_RAYS_DESC dispatch_rays = dispatch_desc;

    dispatch_rays.Width = img_dims.x;
    dispatch_rays.Height = img_dims.y;
    dispatch_rays.Depth = 1;

    return dispatch_rays;
}

bool RTPipeline::has_global_root_sig() const
{
    return rt_global_sig.sig.Get() != nullptr;
}

ID3D12RootSignature *RTPipeline::global_sig()
{
    return rt_global_sig.get();
}

ID3D12StateObject *RTPipeline::operator->()
{
    return get();
}
ID3D12StateObject *RTPipeline::get()
{
    return state.Get();
}

size_t RTPipeline::compute_shader_record_size(const std::wstring &shader) const
{
    size_t shader_size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    // Size of the shader's local root sig, if any
    auto *sig = shader_signature(shader);
    if (sig) {
        shader_size += sig->total_size();
    }
    return align_to(shader_size, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
}

Geometry::Geometry(Buffer verts,
                   Buffer indices,
                   Buffer normals,
                   Buffer uvs,
                   D3D12_RAYTRACING_GEOMETRY_FLAGS geom_flags)
    : vertex_buf(verts), index_buf(indices), normal_buf(normals), uv_buf(uvs)
{
    desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    desc.Triangles.VertexBuffer.StartAddress = vertex_buf->GetGPUVirtualAddress();
    desc.Triangles.VertexBuffer.StrideInBytes = sizeof(float) * 3;
    desc.Triangles.VertexCount = vertex_buf.size() / desc.Triangles.VertexBuffer.StrideInBytes;
    desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;

    desc.Triangles.IndexBuffer = index_buf->GetGPUVirtualAddress();
    desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    desc.Triangles.IndexCount = index_buf.size() / sizeof(uint32_t);
    desc.Triangles.Transform3x4 = 0;
    desc.Flags = geom_flags;
}

BottomLevelBVH::BottomLevelBVH(std::vector<Geometry> &geoms,
                               D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS build_flags)
    : build_flags(build_flags), geometries(geoms)
{
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(geom_descs),
                   [](const Geometry &g) { return g.desc; });
}

void BottomLevelBVH::enqeue_build(ID3D12Device5 *device, ID3D12GraphicsCommandList4 *cmd_list)
{
    post_build_info = Buffer::default(device,
                                      sizeof(uint64_t),
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    post_build_info_readback =
        Buffer::readback(device, post_build_info.size(), D3D12_RESOURCE_STATE_COPY_DEST);

    post_build_info_desc.DestBuffer = post_build_info->GetGPUVirtualAddress();
    post_build_info_desc.InfoType =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

    // Determine bound of much memory the accel builder may need and allocate it
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bvh_inputs = {0};
    bvh_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bvh_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bvh_inputs.NumDescs = geom_descs.size();
    bvh_inputs.pGeometryDescs = geom_descs.data();
    bvh_inputs.Flags = build_flags;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {0};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&bvh_inputs, &prebuild_info);

    // The buffer sizes must be aligned to 256 bytes
    prebuild_info.ResultDataMaxSizeInBytes =
        align_to(prebuild_info.ResultDataMaxSizeInBytes,
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    prebuild_info.ScratchDataSizeInBytes =
        align_to(prebuild_info.ScratchDataSizeInBytes,
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

#if 0
	std::cout << "TriangleMesh BVH will use at most "
		<< pretty_print_count(prebuild_info.ResultDataMaxSizeInBytes) << "b, and scratch of: "
		<< pretty_print_count(prebuild_info.ScratchDataSizeInBytes) << "b\n";
#endif

    bvh = Buffer::default(device,
                          prebuild_info.ResultDataMaxSizeInBytes,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    scratch = Buffer::default(device,
                              prebuild_info.ScratchDataSizeInBytes,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {0};
    build_desc.Inputs = bvh_inputs;
    build_desc.DestAccelerationStructureData = bvh->GetGPUVirtualAddress();
    build_desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 1, &post_build_info_desc);

    // Insert a barrier to wait for the build to complete, and transition the post build
    // info write buffer to copy source so we can read it back
    std::array<D3D12_RESOURCE_BARRIER, 2> barriers = {
        barrier_uav(bvh),
        barrier_transition(post_build_info, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    cmd_list->ResourceBarrier(barriers.size(), barriers.data());

    // Enqueue a copy of the post-build info to CPU visible memory
    cmd_list->CopyResource(post_build_info_readback.get(), post_build_info.get());
}

void BottomLevelBVH::enqueue_compaction(ID3D12Device5 *device,
                                        ID3D12GraphicsCommandList4 *cmd_list)
{
    uint64_t *map = static_cast<uint64_t *>(post_build_info_readback.map());
    const uint64_t compacted_size =
        align_to(*map, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
#if 0
	std::cout << "Bottom level AS compacted size will be: " << pretty_print_count(compacted_size) << "b\n";
#endif

    post_build_info_readback.unmap();

    if (build_flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) {
        scratch = Buffer::default(device,
                                  compacted_size,
                                  D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        cmd_list->CopyRaytracingAccelerationStructure(
            scratch->GetGPUVirtualAddress(),
            bvh->GetGPUVirtualAddress(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        D3D12_RESOURCE_BARRIER barrier = barrier_uav(scratch);
        cmd_list->ResourceBarrier(1, &barrier);
    }
}

void BottomLevelBVH::finalize()
{
    if (build_flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) {
        bvh = scratch;
    }
    // Release the buffers we don't need anymore
    scratch = Buffer();
    post_build_info = Buffer();
    post_build_info_readback = Buffer();
}

ID3D12Resource *BottomLevelBVH::operator->()
{
    return get();
}
ID3D12Resource *BottomLevelBVH::get()
{
    return bvh.get();
}

TopLevelBVH::TopLevelBVH(Buffer instance_buf,
                         const std::vector<Instance> &instances,
                         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS build_flags)
    : instances(instances), instance_buf(instance_buf), build_flags(build_flags)
{
}

void TopLevelBVH::enqeue_build(ID3D12Device5 *device, ID3D12GraphicsCommandList4 *cmd_list)
{
    // Determine bound of much memory the accel builder may need and allocate it
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bvh_inputs = {0};
    bvh_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    bvh_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bvh_inputs.NumDescs = instances.size();
    bvh_inputs.InstanceDescs = instance_buf->GetGPUVirtualAddress();
    bvh_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {0};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&bvh_inputs, &prebuild_info);

    // The buffer sizes must be aligned to 256 bytes
    prebuild_info.ResultDataMaxSizeInBytes =
        align_to(prebuild_info.ResultDataMaxSizeInBytes,
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    prebuild_info.ScratchDataSizeInBytes =
        align_to(prebuild_info.ScratchDataSizeInBytes,
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
#if 0
	std::cout << "TopLevelBVH will use at most "
		<< pretty_print_count(prebuild_info.ResultDataMaxSizeInBytes) << "b, and scratch of: "
		<< pretty_print_count(prebuild_info.ScratchDataSizeInBytes) << "b\n";
#endif

    bvh = Buffer::default(device,
                          prebuild_info.ResultDataMaxSizeInBytes,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    scratch = Buffer::default(device,
                              prebuild_info.ScratchDataSizeInBytes,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {0};
    build_desc.Inputs = bvh_inputs;
    build_desc.DestAccelerationStructureData = bvh->GetGPUVirtualAddress();
    build_desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    // Insert a barrier to wait for the build to complete
    D3D12_RESOURCE_BARRIER barrier = barrier_uav(bvh);
    cmd_list->ResourceBarrier(1, &barrier);
}

void TopLevelBVH::finalize()
{
    // Release the buffers we don't need anymore
    scratch = Buffer();
}

size_t TopLevelBVH::num_instances() const
{
    return instances.size();
}

ID3D12Resource *TopLevelBVH::operator->()
{
    return get();
}

ID3D12Resource *TopLevelBVH::get()
{
    return bvh.get();
}

}
