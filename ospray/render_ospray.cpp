#include <iostream>
#include "render_ospray.h"

RenderOSPRay::RenderOSPRay() : fb(nullptr) {
	const char *argv[] = {"render_ospray_backend"};
	int argc = 1;
	if (ospInit(&argc, argv) != OSP_NO_ERROR) {
		std::cout << "Failed to init OSPRay\n";
		throw std::runtime_error("Failed to init OSPRay");
	}

	world = ospNewModel();
	camera = ospNewCamera("perspective");
	renderer = ospNewRenderer("ao");
	ospSet1i(renderer, "aoSamples", 4);
#if OSPRAY_VERSION_MAJOR == 1
	ospSetObject(renderer, "model", world);
	ospSetObject(renderer, "camera", camera);
#endif
}

void RenderOSPRay::initialize(const int fb_width, const int fb_height) {
	ospSet1f(camera, "aspect", static_cast<float>(fb_width) / fb_height);

	if (fb) {
		ospRelease(fb);
	}

	fb = ospNewFrameBuffer(osp::vec2i{fb_width, fb_height},
			OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
	img.resize(fb_width * fb_height);
}

void RenderOSPRay::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	OSPData verts_data = ospNewData(verts.size() / 3, OSP_FLOAT3, verts.data());
	ospCommit(verts_data);
	OSPData indices_data = ospNewData(indices.size() / 3, OSP_INT3, indices.data());
	ospCommit(indices_data);

	OSPGeometry geom = ospNewGeometry("triangles");
	ospSetObject(geom, "vertex", verts_data);
	ospSetObject(geom, "index", indices_data);
	ospCommit(geom);

	ospAddGeometry(world, geom);
	ospCommit(world);
}

double RenderOSPRay::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	ospSet3fv(camera, "pos", &pos.x);
	ospSet3fv(camera, "dir", &dir.x);
	ospSet3fv(camera, "up", &up.x);
	ospSet1f(camera, "fovy", fovy);
	ospCommit(camera);

#if OSPRAY_VERSION_MAJOR == 2
	if (camera_changed) {
		ospResetAccumulation(fb);
	}
	ospRenderFrame(fb, renderer, camera, world);
#else
	if (camera_changed) {
		ospCommit(renderer);
		ospFrameBufferClear(fb, OSP_FB_COLOR | OSP_FB_ACCUM);
	}
	ospRenderFrame(fb, renderer, OSP_FB_COLOR);
#endif

	const uint32_t *mapped = static_cast<const uint32_t*>(ospMapFrameBuffer(fb, OSP_FB_COLOR));
	std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
	ospUnmapFrameBuffer(mapped, fb);

	return -1;
}

