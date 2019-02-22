#include <limits>
#include "render_embree.h"

static float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
}

RenderEmbree::RenderEmbree()
	: device(rtcNewDevice(NULL)), scene(rtcNewScene(device))
{
}

void RenderEmbree::initialize(const int fb_width, const int fb_height) {
	fb_dims = glm::ivec2(fb_width, fb_height);
	img.resize(fb_width * fb_height);
}

void RenderEmbree::set_mesh(const std::vector<float> &verts,
		const std::vector<uint32_t> &indices)
{
	RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	RTCBuffer vert_buf = rtcNewBuffer(device, verts.size() * sizeof(float));
	RTCBuffer index_buf = rtcNewBuffer(device, indices.size() * sizeof(uint32_t));

	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
			vert_buf, 0, 16, verts.size() / 3);
	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
			index_buf, 0, 12, indices.size() / 3);

	std::copy(verts.begin(), verts.end(),
			static_cast<float*>(rtcGetBufferData(vert_buf)));
	std::copy(indices.begin(), indices.end(),
			static_cast<uint32_t*>(rtcGetBufferData(index_buf)));

	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0);

	rtcAttachGeometry(scene, geom);
	rtcCommitScene(scene);
}

void RenderEmbree::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy)
{
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y * static_cast<float>(fb_dims.x) / fb_dims.y;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	RTCIntersectContext context;
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
	context.filter = NULL;

	// TODO: Trace ray streams, parallelize over tiles with TBB,
	// shade and trace from ISPC for vectorization
	for (int j = 0; j < fb_dims.y; ++j) {
		for (int i = 0; i < fb_dims.x; ++i) {
			const glm::vec2 px = glm::vec2(i + 0.5f, j + 0.5f) / glm::vec2(fb_dims);
			const glm::vec3 dir = glm::normalize(px.x * dir_du
					+ px.y * dir_dv + dir_top_left);
			RTCRay ray;
			ray.org_x = pos.x;
			ray.org_y = pos.y;
			ray.org_z = pos.z;

			ray.dir_x = dir.x;
			ray.dir_y = dir.y;
			ray.dir_z = dir.z;

			ray.tnear = 0.f;
			ray.tfar = std::numeric_limits<float>::infinity();
			ray.time = 0.f;

			ray.mask = std::numeric_limits<uint32_t>::max();
			ray.id = 0;
			ray.flags = 0;

			RTCRayHit ray_hit;
			std::memset(&ray_hit, 0, sizeof(RTCRayHit));
			ray_hit.ray = ray;

			rtcIntersect1(scene, &context, &ray_hit);

			if (ray_hit.hit.geomID != std::numeric_limits<uint32_t>::max()) {
				img[j * fb_dims.x + i] = std::numeric_limits<uint32_t>::max();
			}
		}
	}
}

