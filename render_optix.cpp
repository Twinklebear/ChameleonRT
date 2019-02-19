#include "render_optix.h"

extern "C" const char render_optix_programs[];

using namespace optix;

RenderOptiX::RenderOptiX() {
	context = Context::create();
	context->setRayTypeCount(1);
	context->setEntryPointCount(1);
	Program prog = context->createProgramFromPTXString(render_optix_programs, "perspective_camera");
	context->setRayGenerationProgram(0, prog);
}

void RenderOptiX::initialize(const float fovy,
		const int fb_width, const int fb_height)
{
	width = fb_width;
	height = fb_height;

	fb = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4,
			fb_width, fb_height);
	context["framebuffer"]->setBuffer(fb);
	img.resize(fb_width * fb_height);
}
void RenderOptiX::set_mesh(const std::vector<float> &verts,
		const std::vector<int32_t> &indices)
{
}
void RenderOptiX::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up)
{
	context->launch(0, width, height);
	const uint32_t *mapped = static_cast<const uint32_t*>(fb->map(0, RT_BUFFER_MAP_READ));

	std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
	fb->unmap();
}

