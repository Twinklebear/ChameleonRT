#include <limits>
#include <algorithm>
#include <glm/ext.hpp>
#include "embree_utils.h"

namespace embree {

TriangleMesh::TriangleMesh(RTCDevice &device,
		const std::vector<glm::vec3> &verts,
		const std::vector<glm::uvec3> &indices,
		const std::vector<glm::vec3> &normals,
		const std::vector<glm::vec2> &uvs)
	: geom(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)),
	scene(rtcNewScene(device)),
	index_buf(indices),
	normal_buf(normals),
	uv_buf(uvs)
{
	vertex_buf.reserve(verts.size());
	std::transform(verts.begin(), verts.end(),
			std::back_inserter(vertex_buf),
			[](const glm::vec3 &v) { return glm::vec4(v, 0.f); });

	vbuf = rtcNewSharedBuffer(device, vertex_buf.data(),
			vertex_buf.size() * sizeof(glm::vec4));
	ibuf = rtcNewSharedBuffer(device, index_buf.data(),
			index_buf.size() * sizeof(glm::uvec3));

	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
			vbuf, 0, sizeof(glm::vec4), vertex_buf.size());
	rtcSetGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
			ibuf, 0, sizeof(glm::uvec3), index_buf.size());

	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
	rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0);
	rtcCommitGeometry(geom);

	rtcAttachGeometry(scene, geom);
	rtcCommitScene(scene);
}

TriangleMesh::~TriangleMesh() {
	if (geom) {
		rtcReleaseGeometry(geom);
		rtcReleaseScene(scene);
		rtcReleaseBuffer(vbuf);
		rtcReleaseBuffer(ibuf);
	}
}

RTCScene TriangleMesh::handle() {
	return scene;
}

Instance::Instance(RTCDevice &device, std::shared_ptr<TriangleMesh> &mesh,
		uint32_t material_id,
		const glm::mat4 &transform)
	: handle(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE)),
	mesh(mesh),
	transform(transform),
	inv_transform(glm::inverse(inv_transform)),
	material_id(material_id)
{
	rtcSetGeometryInstancedScene(handle, mesh->handle());
	rtcSetGeometryTransform(handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
			glm::value_ptr(transform));
	rtcCommitGeometry(handle);
}

Instance::~Instance() {
	if (handle) {
		rtcReleaseGeometry(handle);
	}
}

Instance::Instance(Instance &&i)
	: handle(i.handle), mesh(i.mesh)
{
	i.mesh = nullptr;
	i.handle = 0;
}

Instance& Instance::operator=(Instance &&i) {
	if (handle) {
		rtcReleaseGeometry(handle);
	}
	handle = i.handle;
	mesh = i.mesh;

	i.handle = 0;
	i.mesh = nullptr;
	return *this;
}

ISPCInstance::ISPCInstance(const Instance &instance)
	: vertex_buf(instance.mesh->vertex_buf.data()),
	index_buf(instance.mesh->index_buf.data()),
	transform(glm::value_ptr(instance.transform)),
	inv_transform(glm::value_ptr(instance.inv_transform)),
	material_id(instance.material_id)
{
	if (!instance.mesh->normal_buf.empty()) {
		normal_buf = instance.mesh->normal_buf.data();
	}
	if (!instance.mesh->uv_buf.empty()) {
		uv_buf = instance.mesh->uv_buf.data();
	}
}

TopLevelBVH::TopLevelBVH(RTCDevice &device, std::vector<Instance> inst)
	: handle(rtcNewScene(device)),
	instances(std::move(inst))
{
	for (const auto &i : instances) {
		rtcAttachGeometry(handle, i.handle);
		ispc_instances.emplace_back(i);
	}
	rtcCommitScene(handle);
}

TopLevelBVH::~TopLevelBVH() {
	if (handle) {
		rtcReleaseScene(handle);
	}
}

ISPCTexture2D::ISPCTexture2D(const Image &img)
	: width(img.width), height(img.height), channels(img.channels), data(img.img.data())
{}

RaySoA::RaySoA(const size_t nrays)
	: org_x(nrays, 0.f), org_y(nrays, 0.f), org_z(nrays, 0.f), tnear(nrays, 0.f),
	dir_x(nrays, 0.f), dir_y(nrays, 0.f), dir_z(nrays, 0.f), time(nrays, 0.f),
	tfar(nrays, std::numeric_limits<float>::infinity()),
	mask(nrays, std::numeric_limits<uint32_t>::max()),
	id(nrays, 0),
	flags(nrays, 0)
{}

void RaySoA::resize(const size_t nrays) {
	org_x.resize(nrays, 0.f);
	org_y.resize(nrays, 0.f);
	org_z.resize(nrays, 0.f);
	tnear.resize(nrays, 0.f);
	dir_x.resize(nrays, 0.f);
	dir_y.resize(nrays, 0.f);
	dir_z.resize(nrays, 0.f);
	time.resize(nrays, 0.f);
	tfar.resize(nrays, std::numeric_limits<float>::infinity());
	mask.resize(nrays, std::numeric_limits<uint32_t>::max());
	id.resize(nrays, 0);
	flags.resize(nrays, 0);

}

HitSoA::HitSoA(const size_t nrays)
	: ng_x(nrays, 0.f), ng_y(nrays, 0.f), ng_z(nrays, 0.f),
	u(nrays, 0.f), v(nrays, 0.f),
	prim_id(nrays, std::numeric_limits<uint32_t>::max()),
	geom_id(nrays, std::numeric_limits<uint32_t>::max()),
	inst_id(nrays, std::numeric_limits<uint32_t>::max())
{}

void HitSoA::resize(const size_t nrays) {
	ng_x.resize(nrays, 0.f);
	ng_y.resize(nrays, 0.f);
	ng_z.resize(nrays, 0.f);
	u.resize(nrays, 0.f);
	v.resize(nrays, 0.f);
	prim_id.resize(nrays, std::numeric_limits<uint32_t>::max());
	geom_id.resize(nrays, std::numeric_limits<uint32_t>::max());
	inst_id.resize(nrays, std::numeric_limits<uint32_t>::max());
}

RTCRayHitNp make_ray_hit_soa(RaySoA &rays, HitSoA &hits) {
	RTCRayHitNp rh;
	rh.ray.org_x = rays.org_x.data();
	rh.ray.org_y = rays.org_y.data();
	rh.ray.org_z = rays.org_z.data();
	rh.ray.tnear = rays.tnear.data();

	rh.ray.dir_x = rays.dir_x.data();
	rh.ray.dir_y = rays.dir_y.data();
	rh.ray.dir_z = rays.dir_z.data();
	rh.ray.time = rays.time.data();
	rh.ray.tfar = rays.tfar.data();

	rh.ray.mask = rays.mask.data();
	rh.ray.id = rays.id.data();
	rh.ray.flags = rays.flags.data();

	rh.hit.Ng_x = hits.ng_x.data();
	rh.hit.Ng_y = hits.ng_y.data();
	rh.hit.Ng_z = hits.ng_z.data();

	rh.hit.u = hits.u.data();
	rh.hit.v = hits.v.data();

	rh.hit.primID = hits.prim_id.data();
	rh.hit.geomID = hits.geom_id.data();
	rh.hit.instID[0] = hits.inst_id.data();
	return rh;
}

}

