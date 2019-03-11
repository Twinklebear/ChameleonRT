#include "render_optix.h"

extern "C" const char render_optix_ptx[];

using namespace optix;

RenderOptiX::RenderOptiX() {
	context = Context::create();
	context->setRayTypeCount(1);
	context->setEntryPointCount(1);
	Program prog = context->createProgramFromPTXString(render_optix_ptx, "perspective_camera");
	context->setRayGenerationProgram(0, prog);
}

void RenderOptiX::initialize(const int fb_width, const int fb_height) {
	width = fb_width;
	height = fb_height;

	fb = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4,
			fb_width, fb_height);
	context["framebuffer"]->setBuffer(fb);
	img.resize(fb_width * fb_height);
}

void RenderOptiX::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	const size_t num_verts = verts.size() / 3;
	const size_t num_tris = indices.size() / 3;
	auto vertex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_verts);
	std::copy(verts.begin(), verts.end(), (float*)vertex_buffer->map());
	vertex_buffer->unmap();

	auto index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, num_tris);
	std::copy(indices.begin(), indices.end(), (uint32_t*)index_buffer->map());
	index_buffer->unmap();

	auto geom_tri = context->createGeometryTriangles();
	geom_tri->setPrimitiveCount(num_tris);
	geom_tri->setTriangleIndices(index_buffer, RT_FORMAT_UNSIGNED_INT3);
	geom_tri->setVertices(num_verts, vertex_buffer, RT_FORMAT_FLOAT3);

	auto mat = context->createMaterial();
	mat->setClosestHitProgram(0,
			context->createProgramFromPTXString(render_optix_ptx, "closest_hit"));

	auto instance = context->createGeometryInstance(geom_tri, mat);
	// We use these in the hit program to color by normal
	instance["index_buffer"]->set(index_buffer);
	instance["vertex_buffer"]->set(vertex_buffer);

	auto model = context->createGeometryGroup();
	model->addChild(instance);
	model->setAcceleration(context->createAcceleration("Trbvh"));

	context["model"]->set(model);
}

void RenderOptiX::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(width) / height;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	context["cam_pos"]->setFloat(pos.x, pos.y, pos.z);
	context["cam_du"]->setFloat(dir_du.x, dir_du.y, dir_du.z);
	context["cam_dv"]->setFloat(dir_dv.x, dir_dv.y, dir_dv.z);
	context["cam_dir_top_left"]->setFloat(dir_top_left.x, dir_top_left.y, dir_top_left.z);

	context->launch(0, width, height);
	const uint32_t *mapped = static_cast<const uint32_t*>(fb->map(0, RT_BUFFER_MAP_READ));

	std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
	fb->unmap();
}

