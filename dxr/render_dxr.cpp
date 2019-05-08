#include <iostream>
#include <array>
#include <algorithm>
#include <chrono>
#include "render_dxr.h"
#include "render_dxr_embedded_dxil.h"

#define CHECK_ERR(FN) \
	{ \
		auto res = FN; \
		if (FAILED(res)) { \
			std::cout << #FN << " failed due to " \
				<< std::hex << res << std::endl << std::flush; \
			throw std::runtime_error(#FN); \
		}\
	}\

using Microsoft::WRL::ComPtr;

bool dxr_available(ComPtr<ID3D12Device5> &device) {
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_data = { 0 };
	CHECK_ERR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
		&feature_data, sizeof(feature_data)));

	return feature_data.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}

RenderDXR::RenderDXR() {
	// Enable debugging for D3D12
	ComPtr<ID3D12Debug> debug_controller;
	auto err = D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
	if (FAILED(err)) {
		std::cout << "Failed to enable debug layer!\n";
		throw std::runtime_error("get debug failed");
	}
	debug_controller->EnableDebugLayer();

	err = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
	if (FAILED(err)) {
		std::cout << "Failed to make D3D12 device\n";
		throw std::runtime_error("failed to make d3d12 device\n");
	}

	if (!dxr_available(device)) {
		throw std::runtime_error("DXR is required but not available!");
	}

	device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
	fence_evt = CreateEvent(nullptr, false, false, nullptr);

	// Create the command queue and command allocator
	D3D12_COMMAND_QUEUE_DESC queue_desc = { 0 };
	queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	CHECK_ERR(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&cmd_queue)));
	CHECK_ERR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(&cmd_allocator)));

	// Make the command list
	CHECK_ERR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmd_allocator.Get(),
		nullptr, IID_PPV_ARGS(&cmd_list)));
	CHECK_ERR(cmd_list->Close());

	// Allocate a constants buffer for the view parameters.
	// These are write once, read once (assumed to change each frame).
	// The params will be:
	// vec4 cam_pos
	// vec4 cam_du
	// vec4 cam_dv
	// vec4 cam_dir_top_left
	{
		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_UPLOAD;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = 4 * sizeof(glm::vec4);
		// Buffer size must be aligned
		res_desc.Width += D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT -
			res_desc.Width % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr, IID_PPV_ARGS(&view_param_buf)));
	}

	build_raytracing_pipeline();
	build_shader_resource_heap();
	build_shader_binding_table();
}

RenderDXR::~RenderDXR() {
	CloseHandle(fence_evt);
}

void RenderDXR::initialize(const int fb_width, const int fb_height) {
	if (img_dims.x == fb_width && img_dims.y == fb_height) {
		return;
	}
	img_dims = glm::ivec2(fb_width, fb_height);
	img.resize(img_dims.x * img_dims.y);

	// Allocate the output texture for the renderer
	{
		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_DEFAULT;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		res_desc.Width = img_dims.x;
		res_desc.Height = img_dims.y;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr, IID_PPV_ARGS(&render_target)));
	}

	// Allocate the readback buffer so we can read the image back to the CPU
	{
		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_READBACK;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = img_dims.x * img_dims.y * 4;
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr, IID_PPV_ARGS(&img_readback_buf)));
	}
}

void RenderDXR::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	// Upload the mesh to the vertex buffer, build accel structures
	
	// Place the vertex data in an upload heap first, then do a GPU-side copy
	// into a default heap (resident in VRAM)
	ComPtr<ID3D12Resource> upload_verts, upload_indices;
	{
		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_UPLOAD;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = verts.size() * sizeof(float);
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr, IID_PPV_ARGS(&upload_verts)));

		res_desc.Width = indices.size() * sizeof(uint32_t);
		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr, IID_PPV_ARGS(&upload_indices)));

		// Allocate the vertex and index buffers in the default heap (GPU memory)
		props.Type = D3D12_HEAP_TYPE_DEFAULT;
		res_desc.Width = verts.size() * sizeof(float);
		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr, IID_PPV_ARGS(&vertex_buf)));

		res_desc.Width = indices.size() * sizeof(uint32_t);
		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr, IID_PPV_ARGS(&index_buf)));
	}
	{
		// Copy over the vertex data
		void *mapping = nullptr;
		D3D12_RANGE range = { 0 };
		CHECK_ERR(upload_verts->Map(0, &range, &mapping));
		std::memcpy(mapping, verts.data(), sizeof(float) * verts.size());
		upload_verts->Unmap(0, nullptr);
	}
	{
		// Copy over the index data
		void *mapping = nullptr;
		D3D12_RANGE range = { 0 };
		CHECK_ERR(upload_indices->Map(0, &range, &mapping));
		std::memcpy(mapping, indices.data(), sizeof(uint32_t) * indices.size());
		upload_indices->Unmap(0, nullptr);
	}

	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

	// Enqueue the copy into GPU memory
	cmd_list->CopyResource(vertex_buf.Get(), upload_verts.Get());
	cmd_list->CopyResource(index_buf.Get(), upload_indices.Get());

	// Barriers to wait for the copies to finish before building the accel. structs
	{
		std::array<D3D12_RESOURCE_BARRIER, 2> barriers;
		for (auto &b : barriers) {
			b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		}
		barriers[0].Transition.pResource = vertex_buf.Get();
		barriers[1].Transition.pResource = index_buf.Get();
		cmd_list->ResourceBarrier(barriers.size(), barriers.data());
	}

	// Now we cam build the bottom level acceleration structure on our triangles
	D3D12_RAYTRACING_GEOMETRY_DESC rt_geom_desc = { 0 };
	rt_geom_desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	rt_geom_desc.Triangles.VertexBuffer.StartAddress = vertex_buf->GetGPUVirtualAddress();
	rt_geom_desc.Triangles.VertexBuffer.StrideInBytes = sizeof(float) * 3;
	rt_geom_desc.Triangles.VertexCount = verts.size() / 3;
	rt_geom_desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;

	rt_geom_desc.Triangles.IndexBuffer = index_buf->GetGPUVirtualAddress();
	rt_geom_desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
	rt_geom_desc.Triangles.IndexCount = indices.size();
	rt_geom_desc.Triangles.Transform3x4 = 0;
	rt_geom_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	// Build the bottom level acceleration structure
	ComPtr<ID3D12Resource> bottom_scratch;
	{
		// Determine bound of much memory the accel builder may need and allocate it
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS as_inputs = { 0 };
		as_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		as_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		as_inputs.NumDescs = 1;
		as_inputs.pGeometryDescs = &rt_geom_desc;
		as_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = { 0 };
		device->GetRaytracingAccelerationStructurePrebuildInfo(&as_inputs, &prebuild_info);

		// The buffer sizes must be aligned to 256 bytes
		prebuild_info.ResultDataMaxSizeInBytes += D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT -
			prebuild_info.ResultDataMaxSizeInBytes % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
		prebuild_info.ScratchDataSizeInBytes += D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT -
			prebuild_info.ScratchDataSizeInBytes % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;

		std::cout << "Bottom level AS will use at most " << prebuild_info.ResultDataMaxSizeInBytes
			<< " bytes, and scratch of " << prebuild_info.ScratchDataSizeInBytes << " bytes\n";

		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_DEFAULT;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = prebuild_info.ResultDataMaxSizeInBytes;
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
			nullptr, IID_PPV_ARGS(&bottom_level_as)));

		res_desc.Width = prebuild_info.ScratchDataSizeInBytes;
		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr, IID_PPV_ARGS(&bottom_scratch)));

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = { 0 };
		build_desc.Inputs = as_inputs;
		build_desc.DestAccelerationStructureData = bottom_level_as->GetGPUVirtualAddress();
		build_desc.ScratchAccelerationStructureData = bottom_scratch->GetGPUVirtualAddress();
		cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, NULL);

		// Insert a barrier to wait for the bottom level AS to complete before
		// we start the top level build
		D3D12_RESOURCE_BARRIER build_barrier = { 0 };
		build_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		build_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		build_barrier.UAV.pResource = bottom_level_as.Get();
		cmd_list->ResourceBarrier(1, &build_barrier);
	}

	// The top-level AS is built over the instances of our bottom level AS
	// in the scene. For now we just have 1, with an identity transform
	{
		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_UPLOAD;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
		// Align it to the required size
		res_desc.Width += D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT -
			res_desc.Width % D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT;
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr, IID_PPV_ARGS(&instance_buf)));

		// Write the data about our instance
		D3D12_RAYTRACING_INSTANCE_DESC *buf;
		instance_buf->Map(0, nullptr, reinterpret_cast<void**>(&buf));

		buf->InstanceID = 0;
		// TODO: does this mean you can do per-instance hit shaders? (yes I think so)
		buf->InstanceContributionToHitGroupIndex = 0;
		buf->Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
		buf->AccelerationStructure = bottom_level_as->GetGPUVirtualAddress();
		buf->InstanceMask = 0xff;

		// Note: D3D matrices are row-major
		std::memset(buf->Transform, 0, sizeof(buf->Transform));
		buf->Transform[0][0] = 1.f;
		buf->Transform[1][1] = 1.f;
		buf->Transform[2][2] = 1.f;

		instance_buf->Unmap(0, nullptr);
	}

	// Now build the top level acceleration structure on our instance
	ComPtr<ID3D12Resource> top_scratch;
	{
		// Determine bound of much memory the accel builder may need and allocate it
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS as_inputs = { 0 };
		as_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
		as_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		as_inputs.NumDescs = 1;
		as_inputs.InstanceDescs = instance_buf->GetGPUVirtualAddress();
		as_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = { 0 };
		device->GetRaytracingAccelerationStructurePrebuildInfo(&as_inputs, &prebuild_info);

		// The buffer sizes must be aligned to 256 bytes
		prebuild_info.ResultDataMaxSizeInBytes += D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT -
			prebuild_info.ResultDataMaxSizeInBytes % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
		prebuild_info.ScratchDataSizeInBytes += D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT -
			prebuild_info.ScratchDataSizeInBytes % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;

		std::cout << "Top level AS will use at most " << prebuild_info.ResultDataMaxSizeInBytes
			<< " bytes, and scratch of " << prebuild_info.ScratchDataSizeInBytes << " bytes\n";

		D3D12_HEAP_PROPERTIES props = { 0 };
		props.Type = D3D12_HEAP_TYPE_DEFAULT;
		props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
		props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

		D3D12_RESOURCE_DESC res_desc = { 0 };
		res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		res_desc.Width = prebuild_info.ResultDataMaxSizeInBytes;
		res_desc.Height = 1;
		res_desc.DepthOrArraySize = 1;
		res_desc.MipLevels = 1;
		res_desc.Format = DXGI_FORMAT_UNKNOWN;
		res_desc.SampleDesc.Count = 1;
		res_desc.SampleDesc.Quality = 0;
		res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
			nullptr, IID_PPV_ARGS(&top_level_as)));

		res_desc.Width = prebuild_info.ScratchDataSizeInBytes;
		CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
			&res_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr, IID_PPV_ARGS(&top_scratch)));

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = { 0 };
		build_desc.Inputs = as_inputs;
		build_desc.DestAccelerationStructureData = top_level_as->GetGPUVirtualAddress();
		build_desc.ScratchAccelerationStructureData = top_scratch->GetGPUVirtualAddress();
		cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, NULL);

		// Insert a barrier to wait for the top level AS to complete
		D3D12_RESOURCE_BARRIER build_barrier = { 0 };
		build_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		build_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		build_barrier.UAV.pResource = top_level_as.Get();
		cmd_list->ResourceBarrier(1, &build_barrier);
	}

	CHECK_ERR(cmd_list->Close());
	std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());

	// Wait for the work to finish
	sync_gpu();
}

double RenderDXR::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	using namespace std::chrono;

	update_view_parameters(pos, dir, up, fovy);
	// Set the render target and TLAS pointers in the descriptor heap
	update_descriptor_heap();

	// Now render!
	CHECK_ERR(cmd_allocator->Reset());
	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

	// Tell the dispatch rays how we built our shader binding table
	D3D12_DISPATCH_RAYS_DESC dispatch_rays = { 0 };
	// RayGen is first, and has a shader identifier and one param
	dispatch_rays.RayGenerationShaderRecord.StartAddress = shader_table->GetGPUVirtualAddress();
	dispatch_rays.RayGenerationShaderRecord.SizeInBytes = 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

	// Miss is next, followed by hit, each is just a shader identifier
	dispatch_rays.MissShaderTable.StartAddress =
		shader_table->GetGPUVirtualAddress() + 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	dispatch_rays.MissShaderTable.SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	dispatch_rays.MissShaderTable.StrideInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

	dispatch_rays.HitGroupTable.StartAddress =
		shader_table->GetGPUVirtualAddress() + 4 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	dispatch_rays.HitGroupTable.SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	dispatch_rays.HitGroupTable.StrideInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

	dispatch_rays.Width = img_dims.x;
	dispatch_rays.Height = img_dims.y;
	dispatch_rays.Depth = 1;

	cmd_list->SetDescriptorHeaps(1, shader_desc_heap.GetAddressOf());
	cmd_list->SetPipelineState1(rt_state_object.Get());
	cmd_list->DispatchRays(&dispatch_rays);
	
	// We want to just time the raytracing work
	CHECK_ERR(cmd_list->Close());
	std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
	auto start = high_resolution_clock::now();
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());

	// Wait for rendering to finish
	sync_gpu();

	auto end = high_resolution_clock::now();
	const double render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-9;

	// Now copy the rendered image into our readback heap so we can give it back
	// to our simple window to blit the image (TODO: Maybe in the future keep this on the GPU?
	// would we be able to share with GL or need a separate DX window backend?)
	CHECK_ERR(cmd_allocator->Reset());
	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));
	{
		// Render target from UA -> Copy Source
		D3D12_RESOURCE_BARRIER res_barrier;
		res_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		res_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		res_barrier.Transition.pResource = render_target.Get();
		res_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		res_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		res_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		cmd_list->ResourceBarrier(1, &res_barrier);
	}

	{
		// Copy the rendered image to the readback buf so we can access it on the CPU
		D3D12_TEXTURE_COPY_LOCATION dst_desc = { 0 };
		dst_desc.pResource = img_readback_buf.Get();
		dst_desc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		dst_desc.PlacedFootprint.Offset = 0;
		dst_desc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		dst_desc.PlacedFootprint.Footprint.Width = img_dims.x;
		dst_desc.PlacedFootprint.Footprint.Height = img_dims.y;
		dst_desc.PlacedFootprint.Footprint.Depth = 1;
		dst_desc.PlacedFootprint.Footprint.RowPitch = img_dims.x * 4;

		D3D12_TEXTURE_COPY_LOCATION src_desc = { 0 };
		src_desc.pResource = render_target.Get();
		src_desc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		src_desc.SubresourceIndex = 0;

		D3D12_BOX region = { 0 };
		region.left = 0;
		region.right = img_dims.x;
		region.top = 0;
		region.bottom = img_dims.y;
		region.front = 0;
		region.back = 1;
		cmd_list->CopyTextureRegion(&dst_desc, 0, 0, 0, &src_desc, &region);
	}

	// Transition the render target back to UA so we can write to it in the next frame
	{
		D3D12_RESOURCE_BARRIER res_barrier;
		res_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		res_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		res_barrier.Transition.pResource = render_target.Get();
		res_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
		res_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		res_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		cmd_list->ResourceBarrier(1, &res_barrier);
	}

	// Run the copy and wait for it to finish
	CHECK_ERR(cmd_list->Close());
	cmd_lists[0] = cmd_list.Get();
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
	sync_gpu();

	// Map the readback buf and copy out the rendered image
	{
		void *buf = nullptr;
		D3D12_RANGE range = { 0 };
		// Explicitly note we want the whole range to silence debug layer warnings
		range.End = img.size() * sizeof(uint32_t);
		img_readback_buf->Map(0, &range, &buf);
		std::memcpy(img.data(), buf, img.size() * sizeof(uint32_t));
		img_readback_buf->Unmap(0, nullptr);
	}

	return img.size() / render_time;
}

void RenderDXR::build_raytracing_pipeline() {
	D3D12_SHADER_BYTECODE bytecode = { 0 };
	bytecode.pShaderBytecode = render_dxr_dxil;
	bytecode.BytecodeLength = sizeof(render_dxr_dxil);

	// Setup the exports for the shader library
	D3D12_DXIL_LIBRARY_DESC shader_lib = { 0 };
	std::vector<D3D12_EXPORT_DESC> exports;
	std::vector<LPCWSTR> shader_exported_fcns;
	const std::vector<std::wstring> export_fcn_names = {
		L"RayGen", L"Miss", L"ClosestHit"
	};
	for (const auto &fn : export_fcn_names) {
		D3D12_EXPORT_DESC shader_export = { 0 };
		shader_export.ExportToRename = nullptr;
		shader_export.Flags = D3D12_EXPORT_FLAG_NONE;
		shader_export.Name = fn.c_str();
		exports.push_back(shader_export);
		shader_exported_fcns.push_back(fn.c_str());
	}
	shader_lib.DXILLibrary = bytecode;
	shader_lib.NumExports = exports.size();
	shader_lib.pExports = exports.data();

	// Build the hit group which uses our shader library
	D3D12_HIT_GROUP_DESC hit_group = { 0 };
	hit_group.HitGroupExport = L"HitGroup";
	hit_group.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
	hit_group.ClosestHitShaderImport = L"ClosestHit";

	// Make the shader config which defines the maximum size in bytes for the ray
	// payload and attribute structures
	D3D12_RAYTRACING_SHADER_CONFIG shader_desc = { 0 };
	// Payload will just be a float4 color + z
	shader_desc.MaxPayloadSizeInBytes = 4 * sizeof(float);
	// Attribute size is just the float2 barycentrics
	shader_desc.MaxAttributeSizeInBytes = 2 * sizeof(float);

	// Create the root signature for our shader library
	// The closest hit and miss shaders don't need one since they
	// don't make use of a local root signature (no reads from buffers/textures)
	{
		std::vector<D3D12_ROOT_PARAMETER> rt_params;
		// The raygen program takes two parameters:
		// the UAV representing the output image buffer
		// the SRV representing the top-level acceleration structure
		D3D12_ROOT_PARAMETER param = { 0 };
		param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		// UAV param for the output image buffer
		D3D12_DESCRIPTOR_RANGE descrip_range_uav = { 0 };
		descrip_range_uav.BaseShaderRegister = 0;
		descrip_range_uav.NumDescriptors = 1;
		descrip_range_uav.RegisterSpace = 0;
		descrip_range_uav.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		descrip_range_uav.OffsetInDescriptorsFromTableStart = 0;

		param.DescriptorTable.NumDescriptorRanges = 1;
		param.DescriptorTable.pDescriptorRanges = &descrip_range_uav;
		rt_params.push_back(param);

		// SRV for the top-level acceleration structure
		D3D12_DESCRIPTOR_RANGE descrip_range_srv = { 0 };
		descrip_range_srv.BaseShaderRegister = 0;
		descrip_range_srv.NumDescriptors = 1;
		descrip_range_srv.RegisterSpace = 0;
		descrip_range_srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		// Second entry in our table
		descrip_range_srv.OffsetInDescriptorsFromTableStart = 1;

		param.DescriptorTable.pDescriptorRanges = &descrip_range_srv;
		rt_params.push_back(param);

		// Constants buffer param for the view parameters
		D3D12_DESCRIPTOR_RANGE descrip_range_cbv = { 0 };
		descrip_range_cbv.BaseShaderRegister = 0;
		descrip_range_cbv.NumDescriptors = 1;
		descrip_range_cbv.RegisterSpace = 0;
		descrip_range_cbv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
		descrip_range_cbv.OffsetInDescriptorsFromTableStart = 2;

		param.DescriptorTable.pDescriptorRanges = &descrip_range_cbv;
		rt_params.push_back(param);

		D3D12_ROOT_SIGNATURE_DESC root_desc = { 0 };
		root_desc.NumParameters = rt_params.size();
		root_desc.pParameters = rt_params.data();
		// RT root signatures are local (TODO WILL to what? the hit group?)
		root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;

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
			signature_blob->GetBufferSize(), IID_PPV_ARGS(&root_signature)));
	}

	// TODO WILL: In a utility library the ray tracing shader and pipeline object
	// should be handled separately

	// Now we can build the raytracing pipeline. It's made of a bunch of subobjects that
	// describe the shader code libraries, hit groups, root signature associations and
	// some other config stuff
	std::vector<D3D12_STATE_SUBOBJECT> subobjects;
	subobjects.resize(7);
	size_t current_subobj = 0;
	{
		D3D12_STATE_SUBOBJECT dxil_libs = { 0 };
		dxil_libs.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
		dxil_libs.pDesc = &shader_lib;
		subobjects[current_subobj++] = dxil_libs;
	}
	{
		D3D12_STATE_SUBOBJECT hit_grp_obj = { 0 };
		hit_grp_obj.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
		hit_grp_obj.pDesc = &hit_group;
		subobjects[current_subobj++] = hit_grp_obj;
	}
	{
		D3D12_STATE_SUBOBJECT shader_cfg = { 0 };
		shader_cfg.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
		shader_cfg.pDesc = &shader_desc;
		subobjects[current_subobj++] = shader_cfg;
	}

	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shader_paylod_assoc = { 0 };
	shader_paylod_assoc.NumExports = shader_exported_fcns.size();
	shader_paylod_assoc.pExports = shader_exported_fcns.data();
	// Associate with the raytracing shader config subobject
	shader_paylod_assoc.pSubobjectToAssociate = &subobjects[current_subobj - 1];
	{
		D3D12_STATE_SUBOBJECT payload_subobj = { 0 };
		payload_subobj.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
		payload_subobj.pDesc = &shader_paylod_assoc;
		subobjects[current_subobj++] = payload_subobj;
	}

	// The root signature needs two subobjects: one to declare it, and one to associate it
	// with a set of symbols
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION root_sig_assoc = { 0 };
	D3D12_LOCAL_ROOT_SIGNATURE rt_local_root_sig;
	rt_local_root_sig.pLocalRootSignature = root_signature.Get();
	{
		// Declare the root signature
		D3D12_STATE_SUBOBJECT root_sig_obj = { 0 };
		root_sig_obj.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
		root_sig_obj.pDesc = &rt_local_root_sig;
		subobjects[current_subobj++] = root_sig_obj;

		root_sig_assoc.NumExports = 1;
		root_sig_assoc.pExports = &shader_exported_fcns[0];
		root_sig_assoc.pSubobjectToAssociate = &subobjects[current_subobj - 1];

		// Associate it with the symbols
		D3D12_STATE_SUBOBJECT root_assoc = { 0 };
		root_assoc.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
		root_assoc.pDesc = &root_sig_assoc;
		subobjects[current_subobj++] = root_assoc;
	}

	// Add a subobject for the ray tracing pipeline configuration
	D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_cfg = { 0 };
	pipeline_cfg.MaxTraceRecursionDepth = 1;

	// Add to the subobjects
	{
		D3D12_STATE_SUBOBJECT pipeline_subobj = { 0 };
		pipeline_subobj.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
		pipeline_subobj.pDesc = &pipeline_cfg;
		subobjects[current_subobj++] = pipeline_subobj;
	}

	// Describe the set of subobjects in our raytracing pipeline
	D3D12_STATE_OBJECT_DESC pipeline_desc = { 0 };
	pipeline_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
	pipeline_desc.NumSubobjects = current_subobj;
	pipeline_desc.pSubobjects = subobjects.data();

	CHECK_ERR(device->CreateStateObject(&pipeline_desc, IID_PPV_ARGS(&rt_state_object)));
}

void RenderDXR::build_shader_resource_heap() {
	// The resource heap has the pointers/views things to our output image buffer
	// and the top level acceleration structure
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc = { 0 };
	heap_desc.NumDescriptors = 3;
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	CHECK_ERR(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&shader_desc_heap)));
}

void RenderDXR::build_shader_binding_table() {
	// The shader binding table is a table of pointers to the shader code
	// and their respective descriptor heaps
	// An SBT entry is the program ID along with a set of params for the program.
	// the params are either 8 byte pointers (or the example mentions 4 byte constants, how to set or use those?)
	// Furthermore, the stride between elements is specified per-group (ray gen, hit, miss, etc) so it
	// must be padded to the largest size required by any individual entry. Note the order also matters for
	// and should match the instance contribution to hit group index.
	// In this example it's simple: our ray gen program takes a single ptr arg to the rt_shader_res_heap,
	// and our others don't take arguments at all
	// 3 shaders and one that takes a single pointer param (ray-gen). However, each shader
	// binding table in the dispatch rays must have its address start at a 64byte alignment,
	// and use a 32byte stride. So pad these out to meet those requirements by making each
	// entry 64 bytes
	uint32_t sbt_table_size = 4 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	// What's the alignment requirement here?
	sbt_table_size += D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT -
		sbt_table_size % D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
	std::cout << "SBT is " << sbt_table_size << " bytes\n";

	ID3D12StateObjectProperties *rt_pipeline_props = nullptr;
	rt_state_object->QueryInterface(&rt_pipeline_props);

	D3D12_HEAP_PROPERTIES props = { 0 };
	props.Type = D3D12_HEAP_TYPE_UPLOAD;
	props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

	D3D12_RESOURCE_DESC res_desc = { 0 };
	res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	res_desc.Width = sbt_table_size;
	res_desc.Height = 1;
	res_desc.DepthOrArraySize = 1;
	res_desc.MipLevels = 1;
	res_desc.Format = DXGI_FORMAT_UNKNOWN;
	res_desc.SampleDesc.Count = 1;
	res_desc.SampleDesc.Quality = 0;
	res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

	// Place the vertex data in an upload heap first, then do a GPU-side copy
	// into a default heap (resident in VRAM)
	CHECK_ERR(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
		&res_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr, IID_PPV_ARGS(&shader_table)));

	// Map the SBT and write our shader data and param info to it
	uint8_t *sbt_map = nullptr;
	CHECK_ERR(shader_table->Map(0, nullptr, reinterpret_cast<void**>(&sbt_map)));

	// First we write the ray-gen shader identifier, followed by the ptr to its descriptor heap
	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"RayGen"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
	sbt_map += D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

	D3D12_GPU_DESCRIPTOR_HANDLE desc_heap_handle = shader_desc_heap->GetGPUDescriptorHandleForHeapStart();
	std::memcpy(sbt_map, &desc_heap_handle.ptr, sizeof(uint64_t));
	// Each entry must start at an alignment of 32bytes, so offset by the required alignment
	sbt_map += D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;

	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"Miss"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
	sbt_map += 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"HitGroup"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

	shader_table->Unmap(0, nullptr);
}

void RenderDXR::update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
	const glm::vec3 &up, const float fovy) {
	// TODO: Some doc mentioned you can also send 4byte constants directly in the shader table
	// having them embedded like that might be nice for the camera parameters, but how
	// does that get setup? How much difference would it make?
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(img_dims.x) / img_dims.y;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	glm::vec4 *buf = nullptr;
	D3D12_RANGE range = { 0 };
	CHECK_ERR(view_param_buf->Map(0, &range, reinterpret_cast<void**>(&buf)));
	buf[0] = glm::vec4(pos, 0.f);
	buf[1] = glm::vec4(dir_du, 0.f);
	buf[2] = glm::vec4(dir_dv, 0.f);
	buf[3] = glm::vec4(dir_top_left, 0.f);
	view_param_buf->Unmap(0, nullptr);
}

void RenderDXR::update_descriptor_heap() {
	D3D12_CPU_DESCRIPTOR_HANDLE heap_handle = shader_desc_heap->GetCPUDescriptorHandleForHeapStart();

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = { 0 };
	uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(render_target.Get(), nullptr, &uav_desc, heap_handle);

	// Write the TLAS after the output image in the heap
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_SHADER_RESOURCE_VIEW_DESC tlas_desc = { 0 };
	tlas_desc.Format = DXGI_FORMAT_UNKNOWN;
	tlas_desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	tlas_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	tlas_desc.RaytracingAccelerationStructure.Location = top_level_as->GetGPUVirtualAddress();
	device->CreateShaderResourceView(nullptr, &tlas_desc, heap_handle);

	// Write the view params constants buffer
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = { 0 };
	cbv_desc.BufferLocation = view_param_buf->GetGPUVirtualAddress();
	cbv_desc.SizeInBytes = 4 * sizeof(glm::vec4);
	// Align size
	cbv_desc.SizeInBytes += D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT -
		cbv_desc.SizeInBytes % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
	device->CreateConstantBufferView(&cbv_desc, heap_handle);
}

void RenderDXR::sync_gpu() {
	// Sync with the fence to wait for the assets to upload
	const uint64_t signal_val = fence_value++;
	CHECK_ERR(cmd_queue->Signal(fence.Get(), signal_val));

	if (fence->GetCompletedValue() < signal_val) {
		CHECK_ERR(fence->SetEventOnCompletion(signal_val, fence_evt));
		WaitForSingleObject(fence_evt, INFINITE);
	}
	++fence_value;
}
