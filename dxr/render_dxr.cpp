#include <d3d12.h>
#include "render_dxr.h"

RenderDXR::RenderDXR() {}

void RenderDXR::initialize(const int fb_width, const int fb_height) {
	img.resize(fb_width * fb_height);
}
void RenderDXR::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
}
double RenderDXR::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	std::fill(img.begin(), img.end(), 255 << 16);
	return 1.f;
}

