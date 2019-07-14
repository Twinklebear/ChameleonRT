#include <glm/ext.hpp>
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
		render_target.linear_row_pitch() * fb_height, D3D12_RESOURCE_STATE_COPY_DEST);
}

void RenderDXR::set_scene(const Scene &scene) {
	frame_id = 0;

	// TODO: We can actually run all these uploads and BVH builds in parallel
	// using multiple command lists, as long as the BVH builds don't need so
	// much build + scratch that we run out of GPU memory.
	for (const auto &mesh : scene.meshes) {
		// Upload the mesh to the vertex buffer, build accel structures
		// Place the data in an upload heap first, then do a GPU-side copy
		// into a default heap (resident in VRAM)
		Buffer upload_verts = Buffer::upload(device.Get(), mesh.vertices.size() * sizeof(glm::vec3),
				D3D12_RESOURCE_STATE_GENERIC_READ);
		Buffer upload_indices = Buffer::upload(device.Get(), mesh.indices.size() * sizeof(glm::uvec3),
				D3D12_RESOURCE_STATE_GENERIC_READ);

		// Copy vertex and index data into the upload buffers
		std::memcpy(upload_verts.map(), mesh.vertices.data(), upload_verts.size());
		std::memcpy(upload_indices.map(), mesh.indices.data(), upload_indices.size());
		upload_verts.unmap();
		upload_indices.unmap();

		Buffer upload_uvs;
		if (!mesh.uvs.empty()) {
			upload_uvs = Buffer::upload(device.Get(), mesh.uvs.size() * sizeof(glm::vec2),
					D3D12_RESOURCE_STATE_GENERIC_READ);
			std::memcpy(upload_uvs.map(), mesh.uvs.data(), upload_uvs.size());
			upload_uvs.unmap();
		}

		// Allocate GPU side buffers for the data so we can have it resident in VRAM
		Buffer vertex_buf = Buffer::default(device.Get(), upload_verts.size(),
				D3D12_RESOURCE_STATE_COPY_DEST);
		Buffer index_buf = Buffer::default(device.Get(), upload_indices.size(),
				D3D12_RESOURCE_STATE_COPY_DEST);

		CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

		// Enqueue the copy into GPU memory
		cmd_list->CopyResource(vertex_buf.get(), upload_verts.get());
		cmd_list->CopyResource(index_buf.get(), upload_indices.get());

		Buffer uv_buf;
		if (!mesh.uvs.empty()) {
			uv_buf = Buffer::default(device.Get(), upload_uvs.size(),
				D3D12_RESOURCE_STATE_COPY_DEST);
			cmd_list->CopyResource(uv_buf.get(), upload_uvs.get());
		}

		// Barriers to wait for the copies to finish before building the accel. structs
		{
			std::vector<D3D12_RESOURCE_BARRIER> b;
			b.push_back(barrier_transition(vertex_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
			b.push_back(barrier_transition(index_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
			if (!mesh.uvs.empty()) {
				b.push_back(barrier_transition(uv_buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
			};
			cmd_list->ResourceBarrier(b.size(), b.data());
		}

		meshes.emplace_back(vertex_buf, index_buf, uv_buf);
		meshes.back().enqeue_build(device.Get(), cmd_list.Get());

		// TODO: Some possible perf improvements: We can run all the upload of
		// index data in parallel, and the BVH building in parallel for all the
		// geometries. This should help for some large scenes, though with the assumption
		// that the entire build space for all the bottom level stuff can fit on the GPU.
		// For large scenes it would be best to monitor the available space needed for
		// the queued builds vs. the available GPU memory and then run stuff and compact
		// when we start getting full.
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
		for (size_t i = 0; i < scene.meshes.size(); ++i) {
			// TODO: Need some way to express real instancing one I move off OBJ files and
			// have scenes with actual instances.
			buf[i].InstanceID = scene.meshes[i].material_id;
			// Note: we set the num ray type stride for the hit groups here, I think the
			// other multiplier is for doing some sort of per-geometry shaders
			buf[i].InstanceContributionToHitGroupIndex = i * NUM_RAY_TYPES;
			buf[i].Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
			buf[i].AccelerationStructure = meshes[i]->GetGPUVirtualAddress();
			buf[i].InstanceMask = 0xff;

			// Note: D3D matrices are row-major
			std::memset(buf[i].Transform, 0, sizeof(buf[i].Transform));
			buf[i].Transform[0][0] = 1.0f;
			buf[i].Transform[1][1] = 1.0f;
			buf[i].Transform[2][2] = 1.0f;
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

	// Upload the textures
	for (const auto &t : scene.textures) {
		Texture2D tex = Texture2D::default(device.Get(), glm::uvec2(t.width, t.height),
				D3D12_RESOURCE_STATE_COPY_DEST, DXGI_FORMAT_R8G8B8A8_UNORM);
		Buffer tex_upload = Buffer::upload(device.Get(), tex.linear_row_pitch() * t.height,
				D3D12_RESOURCE_STATE_GENERIC_READ);

		// TODO: Some better texture upload handling here, and readback for handling the row pitch stuff
		if (tex.linear_row_pitch() == t.width * tex.pixel_size()) {
			std::memcpy(tex_upload.map(), t.img.data(), tex_upload.size());
		} else {
			uint8_t *buf = static_cast<uint8_t*>(tex_upload.map());
			for (uint32_t y = 0; y < t.height; ++y) {
				std::memcpy(buf + y * tex.linear_row_pitch(),
						t.img.data() + y * t.width * tex.pixel_size(),
						t.width * tex.pixel_size());
			}
		}
		tex_upload.unmap();

		// TODO: We can upload these textures at once as well
		CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));

		tex.upload(cmd_list.Get(), tex_upload);
		auto b = barrier_transition(tex, D3D12_RESOURCE_STATE_GENERIC_READ);
		cmd_list->ResourceBarrier(1, &b);

		CHECK_ERR(cmd_list->Close());
		cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
		sync_gpu();

		textures.push_back(tex);
	}

	// Upload the material data
	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));
	{
		Buffer mat_upload_buf = Buffer::upload(device.Get(),
				scene.materials.size() * sizeof(DisneyMaterial),
				D3D12_RESOURCE_STATE_GENERIC_READ);
		std::memcpy(mat_upload_buf.map(), scene.materials.data(), mat_upload_buf.size());
		mat_upload_buf.unmap();

		material_param_buf = Buffer::default(device.Get(), mat_upload_buf.size(),
				D3D12_RESOURCE_STATE_COPY_DEST);

		cmd_list->CopyResource(material_param_buf.get(), mat_upload_buf.get());
		auto b = barrier_transition(material_param_buf, D3D12_RESOURCE_STATE_GENERIC_READ);
		cmd_list->ResourceBarrier(1, &b);

		CHECK_ERR(cmd_list->Close());
		cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
		sync_gpu();
	}

	build_shader_resource_heap();
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

	// TODO: We'll need a second desc. heap for the sampler and bind both of them here
	std::array<ID3D12DescriptorHeap*, 2> desc_heaps = {
		raygen_desc_heap.get(), raygen_sampler_heap.get()
	};
	cmd_list->SetDescriptorHeaps(desc_heaps.size(), desc_heaps.data());
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
		auto b = barrier_transition(render_target, D3D12_RESOURCE_STATE_COPY_SOURCE);
		cmd_list->ResourceBarrier(1, &b);

		render_target.readback(cmd_list.Get(), img_readback_buf);
	
		// Transition the render target back to UA so we can write to it in the next frame
		b = barrier_transition(render_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
	if (render_target.linear_row_pitch() == render_target.dims().x * render_target.pixel_size()) {
		std::memcpy(img.data(), img_readback_buf.map(), img_readback_buf.size());
	} else {
		uint8_t *buf = static_cast<uint8_t*>(img_readback_buf.map());
		for (uint32_t y = 0; y < render_target.dims().y; ++y) {
			std::memcpy(img.data() + y * render_target.dims().x,
				buf + y * render_target.linear_row_pitch(),
				render_target.dims().x * render_target.pixel_size());
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
	RootSignature raygen_root_sig = RootSignatureBuilder::local()
		.add_desc_heap("cbv_srv_uav_heap", raygen_desc_heap)
		.add_desc_heap("sampler_heap", raygen_sampler_heap)
		.create(device.Get());

	// Create the root signature for our closest hit function
	RootSignature hitgroup_root_sig = RootSignatureBuilder::local()
		.add_srv("vertex_buf", 0, 1)
		.add_srv("index_buf", 1, 1)
		.add_srv("uv_buf", 2, 1)
		.add_constants("MeshData", 0, 1, 1)
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
	// The CBV/SRV/UAV resource heap has the pointers/views things to our output image buffer
	// and the top level acceleration structure, and any textures
	raygen_desc_heap = DescriptorHeapBuilder()
		.add_uav_range(2, 0, 0)
		.add_srv_range(2, 0, 0)
		.add_cbv_range(1, 0, 0)
		.add_srv_range(!textures.empty() ? textures.size() : 1, 2, 0)
		.create(device.Get());

	raygen_sampler_heap = DescriptorHeapBuilder()
		.add_sampler_range(1, 0, 0)
		.create(device.Get());
}

void RenderDXR::build_shader_binding_table() {
	rt_pipeline.map_shader_table();
	{
		uint8_t *map = rt_pipeline.shader_record(L"RayGen");
		const RootSignature *sig = rt_pipeline.shader_signature(L"RayGen");

		// Is writing the descriptor heap handle actually needed? It seems to not matter
		// if this is written or not
		// TODO: MAybe this is the index in the list of heaps we bind at render time to use?
		// Will it find the sampler heap properly if we just have nothing bound here?
		//D3D12_GPU_DESCRIPTOR_HANDLE desc_heap_handle =
		//	raygen_desc_heap->GetGPUDescriptorHandleForHeapStart();
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

		if (meshes[i].uv_buf.size() != 0) {
			gpu_handle = meshes[i].uv_buf->GetGPUVirtualAddress();
			std::memcpy(map + sig->offset("uv_buf"), &gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
		} else {
			gpu_handle = 0;
			std::memcpy(map + sig->offset("uv_buf"), &gpu_handle, sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
		}

		const uint32_t num_uvs = meshes[i].uv_buf.size();
		std::memcpy(map + sig->offset("MeshData"), &num_uvs, sizeof(uint32_t));
	}
	rt_pipeline.unmap_shader_table();

	CHECK_ERR(cmd_list->Reset(cmd_allocator.Get(), nullptr));
	rt_pipeline.upload_shader_table(cmd_list.Get());
	std::array<ID3D12CommandList*, 1> cmd_lists = { cmd_list.Get() };
	CHECK_ERR(cmd_list->Close());
	cmd_queue->ExecuteCommandLists(cmd_lists.size(), cmd_lists.data());
	sync_gpu();
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
		raygen_desc_heap.cpu_desc_handle();

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
		srv_desc.Buffer.NumElements = material_param_buf.size() / sizeof(DisneyMaterial);
		srv_desc.Buffer.StructureByteStride = sizeof(DisneyMaterial);
		srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		device->CreateShaderResourceView(material_param_buf.get(), &srv_desc, heap_handle);
		heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Write the view params constants buffer
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc = { 0 };
	cbv_desc.BufferLocation = view_param_buf->GetGPUVirtualAddress();
	cbv_desc.SizeInBytes = view_param_buf.size();
	device->CreateConstantBufferView(&cbv_desc, heap_handle);
	heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// Write the SRVs for the textures
	for (auto &t : textures) {
		D3D12_SHADER_RESOURCE_VIEW_DESC tex_desc = { 0 };
		tex_desc.Format = t.pixel_format();
		tex_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		tex_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		tex_desc.Texture2D.MipLevels = 1;
		device->CreateShaderResourceView(t.get(), &tex_desc, heap_handle);
		heap_handle.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Write the sampler to the sampler heap
	D3D12_SAMPLER_DESC sampler_desc = {0};
	sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	sampler_desc.MinLOD = 0;
	sampler_desc.MaxLOD = 0;
	sampler_desc.MipLODBias = 0.0f;
	sampler_desc.MaxAnisotropy = 1;
	device->CreateSampler(&sampler_desc, raygen_sampler_heap.cpu_desc_handle());
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

