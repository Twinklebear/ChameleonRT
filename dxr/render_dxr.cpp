#include <iostream>
#include <array>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <iomanip>
#include "util.h"
#include "render_dxr.h"
#include "render_dxr_embedded_dxil.h"

using Microsoft::WRL::ComPtr;

RenderDXR::RenderDXR() {
	// Enable debugging for D3D12
#ifdef _DEBUG
	{
		ComPtr<ID3D12Debug> debug_controller;
		auto err = D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller));
		if (FAILED(err)) {
			std::cout << "Failed to enable debug layer!\n";
			throw std::runtime_error("get debug failed");
		}
		debug_controller->EnableDebugLayer();
	}
#endif

	auto err = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
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
	view_param_buf = Buffer::upload(device.Get(),
		align_to(4 * sizeof(glm::vec4), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),
		D3D12_RESOURCE_STATE_GENERIC_READ);
	build_raytracing_pipeline();
	build_shader_resource_heap();
}

RenderDXR::~RenderDXR() {
	CloseHandle(fence_evt);
}

void RenderDXR::initialize(const int fb_width, const int fb_height) {
	if (render_target.dims().x == fb_width && render_target.dims().y == fb_height) {
		return;
	}
	img.resize(fb_width * fb_height);

	render_target = Texture2D::default(device.Get(), glm::uvec2(fb_width, fb_height),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, DXGI_FORMAT_R8G8B8A8_UNORM,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	
	// Allocate the readback buffer so we can read the image back to the CPU
	img_readback_buf = Buffer::readback(device.Get(), fb_width * fb_height * 4,
		D3D12_RESOURCE_STATE_COPY_DEST);
}

void RenderDXR::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	// Upload the mesh to the vertex buffer, build accel structures
	// Place the vertex data in an upload heap first, then do a GPU-side copy
	// into a default heap (resident in VRAM)
	Buffer upload_verts = Buffer::upload(device.Get(), verts.size() * sizeof(float),
		D3D12_RESOURCE_STATE_GENERIC_READ);
	Buffer upload_indices = Buffer::upload(device.Get(), indices.size() * sizeof(uint32_t),
		D3D12_RESOURCE_STATE_GENERIC_READ);

	// Copy vertex and index data into the upload buffers
	std::memcpy(upload_verts.map(), verts.data(), upload_verts.size());
	upload_verts.unmap();

	std::memcpy(upload_indices.map(), indices.data(), upload_indices.size());
	upload_indices.unmap();

	// Allocate GPU side buffers for the data so we can have it resident in VRAM
	vertex_buf = Buffer::default(device.Get(), verts.size() * sizeof(float),
		D3D12_RESOURCE_STATE_COPY_DEST);
	index_buf = Buffer::default(device.Get(), indices.size() * sizeof(uint32_t),
		D3D12_RESOURCE_STATE_COPY_DEST);

	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

	// Enqueue the copy into GPU memory
	cmd_list->CopyResource(vertex_buf.get(), upload_verts.get());
	cmd_list->CopyResource(index_buf.get(), upload_indices.get());

	// Barriers to wait for the copies to finish before building the accel. structs
	{
		std::array<D3D12_RESOURCE_BARRIER, 2> barriers = {
			barrier_transition(vertex_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
			barrier_transition(index_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
		};
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
	Buffer bottom_scratch;
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
		prebuild_info.ResultDataMaxSizeInBytes =
			align_to(prebuild_info.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
		prebuild_info.ScratchDataSizeInBytes =
			align_to(prebuild_info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
		std::cout << "Bottom level AS will use at most " << pretty_print_count(prebuild_info.ResultDataMaxSizeInBytes)
			<< "b, and scratch of " << pretty_print_count(prebuild_info.ScratchDataSizeInBytes) << "b\n";

		bottom_level_as = Buffer::default(device.Get(), prebuild_info.ResultDataMaxSizeInBytes,
			D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		bottom_scratch = Buffer::default(device.Get(), prebuild_info.ScratchDataSizeInBytes,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = { 0 };
		build_desc.Inputs = as_inputs;
		build_desc.DestAccelerationStructureData = bottom_level_as->GetGPUVirtualAddress();
		build_desc.ScratchAccelerationStructureData = bottom_scratch->GetGPUVirtualAddress();
		cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, NULL);

		// Insert a barrier to wait for the bottom level AS to complete before
		// we start the top level build
		D3D12_RESOURCE_BARRIER build_barrier = barrier_uav(bottom_level_as);
		cmd_list->ResourceBarrier(1, &build_barrier);
	}

	// The top-level AS is built over the instances of our bottom level AS
	// in the scene. For now we just have 1, with an identity transform
	instance_buf = Buffer::upload(device.Get(),
		align_to(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT),
		D3D12_RESOURCE_STATE_GENERIC_READ);
	{
		// Write the data about our instance
		D3D12_RAYTRACING_INSTANCE_DESC *buf =
			static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(instance_buf.map());

		buf->InstanceID = 0;
		// TODO: does this mean you can do per-instance hit groups? I think so
		buf->InstanceContributionToHitGroupIndex = 0;
		buf->Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
		buf->AccelerationStructure = bottom_level_as->GetGPUVirtualAddress();
		buf->InstanceMask = 0xff;

		// Note: D3D matrices are row-major
		std::memset(buf->Transform, 0, sizeof(buf->Transform));
		buf->Transform[0][0] = 1.f;
		buf->Transform[1][1] = 1.f;
		buf->Transform[2][2] = 1.f;

		instance_buf.unmap();
	}

	// Now build the top level acceleration structure on our instance
	Buffer top_scratch;
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
		prebuild_info.ResultDataMaxSizeInBytes =
			align_to(prebuild_info.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
		prebuild_info.ScratchDataSizeInBytes =
			align_to(prebuild_info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
		std::cout << "Top level AS will use at most " << pretty_print_count(prebuild_info.ResultDataMaxSizeInBytes)
			<< "b, and scratch of " << pretty_print_count(prebuild_info.ScratchDataSizeInBytes) << "b\n";

		top_level_as = Buffer::default(device.Get(), prebuild_info.ResultDataMaxSizeInBytes,
			D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		top_scratch = Buffer::default(device.Get(), prebuild_info.ScratchDataSizeInBytes,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = { 0 };
		build_desc.Inputs = as_inputs;
		build_desc.DestAccelerationStructureData = top_level_as->GetGPUVirtualAddress();
		build_desc.ScratchAccelerationStructureData = top_scratch->GetGPUVirtualAddress();
		cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, NULL);

		// Insert a barrier to wait for the top level AS to complete
		D3D12_RESOURCE_BARRIER build_barrier = barrier_uav(top_level_as);
		cmd_list->ResourceBarrier(1, &build_barrier);
	}

	CHECK_ERR(cmd_list->Close());
	std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());

	// Wait for the work to finish
	sync_gpu();
	
	build_shader_binding_table();
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
	dispatch_rays.RayGenerationShaderRecord.SizeInBytes = shader_table_entry_size;

	// Miss is next, followed by hit, each is just a shader identifier
	dispatch_rays.MissShaderTable.StartAddress =
		shader_table->GetGPUVirtualAddress() + shader_table_entry_size;
	dispatch_rays.MissShaderTable.SizeInBytes = shader_table_entry_size;
	dispatch_rays.MissShaderTable.StrideInBytes = shader_table_entry_size;

	dispatch_rays.HitGroupTable.StartAddress =
		shader_table->GetGPUVirtualAddress() + 2 * shader_table_entry_size;
	dispatch_rays.HitGroupTable.SizeInBytes = shader_table_entry_size;
	dispatch_rays.HitGroupTable.StrideInBytes = shader_table_entry_size;

	dispatch_rays.Width = render_target.dims().x;
	dispatch_rays.Height = render_target.dims().y;
	dispatch_rays.Depth = 1;

	cmd_list->SetDescriptorHeaps(1, raygen_shader_desc_heap.GetAddressOf());
	cmd_list->SetPipelineState1(rt_pipeline.get());
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
		D3D12_RESOURCE_BARRIER b = barrier_transition(render_target, D3D12_RESOURCE_STATE_COPY_SOURCE);
		cmd_list->ResourceBarrier(1, &b);
	}

	{
		// Copy the rendered image to the readback buf so we can access it on the CPU
		D3D12_TEXTURE_COPY_LOCATION dst_desc = { 0 };
		dst_desc.pResource = img_readback_buf.get();
		dst_desc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		dst_desc.PlacedFootprint.Offset = 0;
		dst_desc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		dst_desc.PlacedFootprint.Footprint.Width = render_target.dims().x;
		dst_desc.PlacedFootprint.Footprint.Height = render_target.dims().y;
		dst_desc.PlacedFootprint.Footprint.Depth = 1;
		dst_desc.PlacedFootprint.Footprint.RowPitch = render_target.dims().x * 4;

		D3D12_TEXTURE_COPY_LOCATION src_desc = { 0 };
		src_desc.pResource = render_target.get();
		src_desc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		src_desc.SubresourceIndex = 0;

		D3D12_BOX region = { 0 };
		region.left = 0;
		region.right = render_target.dims().x;
		region.top = 0;
		region.bottom = render_target.dims().y;
		region.front = 0;
		region.back = 1;
		cmd_list->CopyTextureRegion(&dst_desc, 0, 0, 0, &src_desc, &region);
	}

	// Transition the render target back to UA so we can write to it in the next frame
	{
		D3D12_RESOURCE_BARRIER b = barrier_transition(render_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		cmd_list->ResourceBarrier(1, &b);
	}

	// Run the copy and wait for it to finish
	CHECK_ERR(cmd_list->Close());
	cmd_lists[0] = cmd_list.Get();
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
	sync_gpu();

	// Map the readback buf and copy out the rendered image
	std::memcpy(img.data(), img_readback_buf.map(), img_readback_buf.size());
	img_readback_buf.unmap();

	return img.size() / render_time;
}

void RenderDXR::build_raytracing_pipeline() {
	ShaderLibrary shader_library(render_dxr_dxil, sizeof(render_dxr_dxil),
		{ L"RayGen", L"Miss", L"ClosestHit" });

	// Create the root signature for our ray gen shader
	// The raygen program takes three parameters:
	// the UAV to the output image buffer
	// the SRV holding the top-level acceleration structure
	// the CBV holding the camera params
	raygen_root_sig = RootSignatureBuilder::local()
		.add_uav_range(1, 0, 0, 0)
		.add_srv_range(1, 0, 0, 1)
		.add_cbv_range(1, 0, 0, 2)
		.create(device.Get());

	// Create the root signature for our closest hit function
	hitgroup_root_sig = RootSignatureBuilder::local()
		.add_srv("vertex_buf", 0, 1)
		.add_srv("index_buf", 1, 1)
		.create(device.Get());

	rt_pipeline = RTPipelineBuilder()
		.add_shader_library(shader_library)
		.set_ray_gen(L"RayGen")
		.add_miss_shader(L"Miss")
		.add_hit_group(HitGroup(L"HitGroup", D3D12_HIT_GROUP_TYPE_TRIANGLES, L"ClosestHit"))
		.set_shader_root_sig({ L"RayGen" }, raygen_root_sig)
		.set_shader_root_sig({ L"ClosestHit" }, hitgroup_root_sig)
		.configure_shader_payload(shader_library.export_names(), 4 * sizeof(float), 2 * sizeof(float))
		.set_max_recursion(1)
		.create(device.Get());
}

void RenderDXR::build_shader_resource_heap() {
	// The resource heap has the pointers/views things to our output image buffer
	// and the top level acceleration structure
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc = { 0 };
	heap_desc.NumDescriptors = 3;
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	CHECK_ERR(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&raygen_shader_desc_heap)));
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
	shader_table_entry_size = 2 * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	const uint32_t sbt_table_size = align_to(3 * shader_table_entry_size,
		D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
	shader_table = Buffer::upload(device.Get(), sbt_table_size,
		D3D12_RESOURCE_STATE_GENERIC_READ);
		
	std::cout << "Shader table entry size: " << shader_table_entry_size << "\n"
		<< "SBT is " << sbt_table_size << " bytes\n";

	ID3D12StateObjectProperties *rt_pipeline_props = nullptr;
	rt_pipeline->QueryInterface(&rt_pipeline_props);

	// Map the SBT and write our shader data and param info to it
	uint8_t *sbt_map = static_cast<uint8_t*>(shader_table.map());

	// First we write the ray-gen shader identifier, followed by the ptr to its descriptor heap
	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"RayGen"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
	{
		D3D12_GPU_DESCRIPTOR_HANDLE desc_heap_handle =
			raygen_shader_desc_heap->GetGPUDescriptorHandleForHeapStart();
		// It seems like writing this for the ray gen shader in the table isn't needed?
		std::memcpy(sbt_map + raygen_root_sig.descriptor_table_offset(),
			&desc_heap_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
	}
	// Each entry must start at an alignment of 32bytes, so offset by the required alignment
	sbt_map += shader_table_entry_size;

	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"Miss"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
	// Nothing after for the miss shader, but must pad to alignment
	sbt_map += shader_table_entry_size;

	std::memcpy(sbt_map, rt_pipeline_props->GetShaderIdentifier(L"HitGroup"),
		D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
	{
		D3D12_GPU_VIRTUAL_ADDRESS gpu_handle = vertex_buf->GetGPUVirtualAddress();
		std::memcpy(sbt_map + hitgroup_root_sig.offset("vertex_buf"),
			&gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));

		gpu_handle = index_buf->GetGPUVirtualAddress();
		std::memcpy(sbt_map + hitgroup_root_sig.offset("index_buf"),
			&gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
	}
	shader_table.unmap();
}

void RenderDXR::update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
	const glm::vec3 &up, const float fovy) {
	// TODO: Some doc mentioned you can also send 4byte constants directly in the shader table
	// having them embedded like that might be nice for the camera parameters, but how
	// does that get setup? How much difference would it make?
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y
		* static_cast<float>(render_target.dims().x) / render_target.dims().y;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	glm::vec4 *buf = static_cast<glm::vec4*>(view_param_buf.map());
	buf[0] = glm::vec4(pos, 0.f);
	buf[1] = glm::vec4(dir_du, 0.f);
	buf[2] = glm::vec4(dir_dv, 0.f);
	buf[3] = glm::vec4(dir_top_left, 0.f);
	view_param_buf.unmap();
}

void RenderDXR::update_descriptor_heap() {
	D3D12_CPU_DESCRIPTOR_HANDLE heap_handle =
		raygen_shader_desc_heap->GetCPUDescriptorHandleForHeapStart();

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = { 0 };
	uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(render_target.get(), nullptr, &uav_desc, heap_handle);

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
	cbv_desc.SizeInBytes = view_param_buf.size();
	device->CreateConstantBufferView(&cbv_desc, heap_handle);
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
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
