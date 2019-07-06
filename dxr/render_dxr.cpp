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

#define NUM_RAY_TYPES 2

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
	// uint32_t frame_id
	view_param_buf = Buffer::upload(device.Get(),
		align_to(5 * sizeof(glm::vec4), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT),
		D3D12_RESOURCE_STATE_GENERIC_READ);

	// The material is packed into 4 float4
	// TODO: Eventually this should be on the default heap as well when I take out the little
	// testing UI for tweaking the material data
	const size_t material_size = 4 * sizeof(glm::vec4);
	material_param_buf = Buffer::upload(device.Get(), 3 * material_size,
		D3D12_RESOURCE_STATE_GENERIC_READ);

	build_shader_resource_heap();
}

RenderDXR::~RenderDXR() {
	CloseHandle(fence_evt);
}

void RenderDXR::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
	img.resize(fb_width * fb_height);

	render_target = Texture2D::default(device.Get(), glm::uvec2(fb_width, fb_height),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, DXGI_FORMAT_R8G8B8A8_UNORM,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	accum_buffer = Texture2D::default(device.Get(), glm::uvec2(fb_width, fb_height),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, DXGI_FORMAT_R32G32B32A32_FLOAT,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	
	// Allocate the readback buffer so we can read the image back to the CPU
	img_readback_buf = Buffer::readback(device.Get(),
		align_to(fb_width * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) * fb_height,
		D3D12_RESOURCE_STATE_COPY_DEST);
}

void RenderDXR::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	set_mesh({verts}, {indices});
}

void RenderDXR::set_meshes(const std::vector<std::vector<float>> &all_verts,
		const std::vector<std::vector<uint32_t>> &all_indices)
{
	frame_id = 0;
	for (size_t i = 0; i < all_verts.size(); ++i) {
		auto &verts = all_verts[i];
		auto &indices = all_indices[i];

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

		meshes.emplace_back(vertex_buf, index_buf);
		meshes.back().enqeue_build(device.Get(), cmd_list.Get());

		CHECK_ERR(cmd_list->Close());
		std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
		cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
		sync_gpu();

		CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));
		meshes.back().enqueue_compaction(device.Get(), cmd_list.Get());
		CHECK_ERR(cmd_list->Close());
		cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
		sync_gpu();

		meshes.back().finalize();
	}

	instance_buf = Buffer::upload(device.Get(),
		align_to(meshes.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
			D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT),
		D3D12_RESOURCE_STATE_GENERIC_READ);

	{
		// Write the data about our instance
		D3D12_RAYTRACING_INSTANCE_DESC *buf =
			static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(instance_buf.map());
		for (size_t i = 0; i < meshes.size(); ++i) {
			buf[i].InstanceID = i;
			// Note: we set the num ray type stride for the hit groups here, because
			// the shader table order may be different than the order we add stuff to
			// TLAS, since the wrapper does its own internal ordering/mapping/etc.
			buf[i].InstanceContributionToHitGroupIndex = i * NUM_RAY_TYPES;
			buf[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
			buf[i].AccelerationStructure = meshes[i]->GetGPUVirtualAddress();
			buf[i].InstanceMask = 0xff;

			// Note: D3D matrices are row-major
			std::memset(buf[i].Transform, 0, sizeof(buf[i].Transform));
			buf[i].Transform[0][0] = 1.0f;
			buf[i].Transform[1][1] = 1.0f;
			buf[i].Transform[2][2] = 1.0f;

			// TODO Testing: Scoot the instances apart some
			buf[i].Transform[0][3] = (static_cast<float>(i + 1) / meshes.size() - 0.5f) * 8.f;
		}
		instance_buf.unmap();
	}

	// Now build the top level acceleration structure on our instance
	scene_bvh = TopLevelBVH(instance_buf, meshes.size());

	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));
	scene_bvh.enqeue_build(device.Get(), cmd_list.Get());
	CHECK_ERR(cmd_list->Close());

	std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
	sync_gpu();

	scene_bvh.finalize();

	build_raytracing_pipeline();
	build_shader_binding_table();
}

double RenderDXR::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;

	// TODO: probably just pass frame_id directly
	if (camera_changed) {
		frame_id = 0;
	}

	update_view_parameters(pos, dir, up, fovy);
	// Set the render target and TLAS pointers in the descriptor heap
	update_descriptor_heap();

	// Now render!
	CHECK_ERR(cmd_allocator->Reset());
	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

	cmd_list->SetDescriptorHeaps(1, raygen_shader_desc_heap.GetAddressOf());
	cmd_list->SetPipelineState1(rt_pipeline.get());

	D3D12_DISPATCH_RAYS_DESC dispatch_rays = rt_pipeline.dispatch_rays(render_target.dims());
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
	
	const uint32_t readback_row_pitch = align_to(render_target.dims().x * 4, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
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
		dst_desc.PlacedFootprint.Footprint.RowPitch = readback_row_pitch;

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
	// We may have needed some padding for the readback buffer, so we might have to read
	// row by row.
	if (readback_row_pitch == render_target.dims().x * 4) {
		std::memcpy(img.data(), img_readback_buf.map(), img_readback_buf.size());
	} else {
		uint8_t *buf = static_cast<uint8_t*>(img_readback_buf.map());
		for (uint32_t y = 0; y < render_target.dims().y; ++y) {
			std::memcpy(img.data() + y * render_target.dims().x,
				buf + y * readback_row_pitch, render_target.dims().x * 4);
		}
	}
	img_readback_buf.unmap();
	++frame_id;

	return img.size() / render_time;
}

void RenderDXR::build_raytracing_pipeline() {
	ShaderLibrary shader_library(render_dxr_dxil, sizeof(render_dxr_dxil),
		{ L"RayGen", L"Miss", L"ClosestHit", L"OcclusionHit", L"AOMiss" });

	// Create the root signature for our ray gen shader
	// The raygen program takes three parameters:
	// the UAVs to the output image buffer and accumulation buffer
	// the SRV holding the top-level acceleration structure, and the material params
	// the CBV holding the camera params
	RootSignature raygen_root_sig = RootSignatureBuilder::local()
		.add_uav_range(2, 0, 0, 0)
		.add_srv_range(2, 0, 0, 2)
		.add_cbv_range(1, 0, 0, 4)
		.create(device.Get());

	// Create the root signature for our closest hit function
	RootSignature hitgroup_root_sig = RootSignatureBuilder::local()
		.add_srv("vertex_buf", 0, 1)
		.add_srv("index_buf", 1, 1)
		.create(device.Get());

	RTPipelineBuilder rt_pipeline_builder = RTPipelineBuilder()
		.add_shader_library(shader_library)
		.set_ray_gen(L"RayGen")
		.add_miss_shaders({ L"Miss", L"AOMiss" })
		.set_shader_root_sig({ L"RayGen" }, raygen_root_sig)
		.configure_shader_payload(shader_library.export_names(), 8 * sizeof(float), 2 * sizeof(float))
		.set_max_recursion(1);

	// Setup hit groups and shader root signatures for our instances.
	// For now this is also easy since they all share the same programs and root signatures,
	// but we just need different hitgroups to set the different params for the meshes
	std::vector<std::wstring> hg_names;
	for (size_t i = 0; i < meshes.size(); ++i) {
		const std::wstring hg_name = L"HitGroup_inst" + std::to_wstring(i);
		hg_names.push_back(hg_name);
		const std::wstring og_name = L"OcclusionGroup_inst" + std::to_wstring(i);

		rt_pipeline_builder.add_hit_groups({
				HitGroup(hg_name, D3D12_HIT_GROUP_TYPE_TRIANGLES, L"ClosestHit"),
				HitGroup(og_name, D3D12_HIT_GROUP_TYPE_TRIANGLES, L"OcclusionHit")});
	}
	rt_pipeline_builder.set_shader_root_sig(hg_names, hitgroup_root_sig);

	rt_pipeline = rt_pipeline_builder.create(device.Get());
}

void RenderDXR::build_shader_resource_heap() {
	// The resource heap has the pointers/views things to our output image buffer
	// and the top level acceleration structure
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc = { 0 };
	heap_desc.NumDescriptors = 5;
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	CHECK_ERR(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&raygen_shader_desc_heap)));
}

void RenderDXR::build_shader_binding_table() {
	rt_pipeline.map_shader_table();
	{
		D3D12_GPU_DESCRIPTOR_HANDLE desc_heap_handle =
			raygen_shader_desc_heap->GetGPUDescriptorHandleForHeapStart();

		uint8_t *map = rt_pipeline.shader_record(L"RayGen");
		const RootSignature *sig = rt_pipeline.shader_signature(L"RayGen");

		// Is writing the descriptor heap handle actually needed? It seems to not matter
		// if this is written or not
		//std::memcpy(map + sig->descriptor_table_offset(), &desc_heap_handle,
		//	sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
	}
	for (size_t i = 0; i < meshes.size(); ++i) {
		const std::wstring hg_name = L"HitGroup_inst" + std::to_wstring(i);
		uint8_t *map = rt_pipeline.shader_record(hg_name);
		const RootSignature *sig = rt_pipeline.shader_signature(hg_name);
		D3D12_GPU_VIRTUAL_ADDRESS gpu_handle = meshes[i].vertex_buf->GetGPUVirtualAddress();
		std::memcpy(map + sig->offset("vertex_buf"), &gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));

		gpu_handle = meshes[i].index_buf->GetGPUVirtualAddress();
		std::memcpy(map + sig->offset("index_buf"), &gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
	}

	rt_pipeline.unmap_shader_table();
}

void RenderDXR::set_material(const DisneyMaterial &mat) {
	frame_id = 0;
	const size_t material_size = 4 * sizeof(glm::vec4);
	uint8_t *buf = static_cast<uint8_t*>(material_param_buf.map());
	for (size_t i = 0; i < 3; ++i) {
		DisneyMaterial m = mat;
		switch (i % 3) {
			case 0: m.base_color *= glm::vec3(1.0f, 0.1f, 0.1f);
					break;
			case 1: m.base_color *= glm::vec3(0.1f, 1.0f, 0.1f);
					break;
			case 2: m.base_color *= glm::vec3(0.1f, 0.1f, 1.0f);
					break;
			default: break;
		}
		std::memcpy(buf + material_size * i, &m, sizeof(DisneyMaterial));
	}
	material_param_buf.unmap();
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

	uint8_t *buf = static_cast<uint8_t*>(view_param_buf.map());
	{
		glm::vec4 *vecs = reinterpret_cast<glm::vec4*>(buf);
		vecs[0] = glm::vec4(pos, 0.f);
		vecs[1] = glm::vec4(dir_du, 0.f);
		vecs[2] = glm::vec4(dir_dv, 0.f);
		vecs[3] = glm::vec4(dir_top_left, 0.f);
	}
	{
		uint32_t *fid = reinterpret_cast<uint32_t*>(buf + 4 * sizeof(glm::vec4));
		*fid = frame_id;
	}

	view_param_buf.unmap();
}

void RenderDXR::update_descriptor_heap() {
	D3D12_CPU_DESCRIPTOR_HANDLE heap_handle =
		raygen_shader_desc_heap->GetCPUDescriptorHandleForHeapStart();

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = { 0 };

	// Render target
	uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(render_target.get(), nullptr, &uav_desc, heap_handle);
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// Accum buffer
	device->CreateUnorderedAccessView(accum_buffer.get(), nullptr, &uav_desc, heap_handle);
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// Write the TLAS after the output image in the heap
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC tlas_desc = { 0 };
		tlas_desc.Format = DXGI_FORMAT_UNKNOWN;
		tlas_desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
		tlas_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		tlas_desc.RaytracingAccelerationStructure.Location = scene_bvh->GetGPUVirtualAddress();
		device->CreateShaderResourceView(nullptr, &tlas_desc, heap_handle);
		heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Write the material params buffer view
	{
		D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = { 0 };
		srv_desc.Format = DXGI_FORMAT_UNKNOWN;
		srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv_desc.Buffer.FirstElement = 0;
		srv_desc.Buffer.NumElements = 3;
		srv_desc.Buffer.StructureByteStride = 4 * sizeof(glm::vec4);
		srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		device->CreateShaderResourceView(material_param_buf.get(), &srv_desc, heap_handle);
		heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Write the view params constants buffer
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = { 0 };
	cbv_desc.BufferLocation = view_param_buf->GetGPUVirtualAddress();
	cbv_desc.SizeInBytes = view_param_buf.size();
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
