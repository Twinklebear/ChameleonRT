#include "render_ospray.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <tbb/parallel_for.h>
#include <util.h>
#include <glm/ext.hpp>

RenderOSPRay::RenderOSPRay() : fb(0)
{
    const char *argv[] = {"render_ospray_backend"};
    int argc = 1;
    if (ospInit(&argc, argv) != OSP_NO_ERROR) {
        std::cout << "Failed to init OSPRay\n";
        throw std::runtime_error("Failed to init OSPRay");
    }

    camera = ospNewCamera("perspective");
    renderer = ospNewRenderer("pathtracer");
    ospCommit(renderer);
}

std::string RenderOSPRay::name()
{
    return "OSPRay";
}

void RenderOSPRay::initialize(const int fb_width, const int fb_height)
{
    float aspect = static_cast<float>(fb_width) / fb_height;
    ospSetParam(camera, "aspect", OSP_FLOAT, &aspect);

    if (fb) {
        ospRelease(fb);
    }

    fb = ospNewFrameBuffer(fb_width, fb_height, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
    img.resize(fb_width * fb_height);
}

void RenderOSPRay::set_scene(const Scene &in_scene)
{
    ospResetAccumulation(fb);

    scene = in_scene;

    // Linearize any sRGB textures beforehand, since we don't have fancy sRGB texture interpolation
    // support in hardware
    tbb::parallel_for(size_t(0), scene.textures.size(), [&](size_t i) {
        auto &img = scene.textures[i];
        if (img.color_space == LINEAR) {
            return;
        }
        img.color_space = LINEAR;
        const int convert_channels = std::min(3, img.channels);
        tbb::parallel_for(size_t(0), size_t(img.width) * img.height, [&](size_t px) {
            for (int c = 0; c < convert_channels; ++c) {
                float x = img.img[px * img.channels + c] / 255.f;
                x = srgb_to_linear(x);
                img.img[px * img.channels + c] = glm::clamp(x * 255.f, 0.f, 255.f);
            }
        });
    });

    textures.clear();
    for (const auto &tex : scene.textures) {
        const OSPDataType data_type = tex.channels == 3 ? OSP_VEC3UC : OSP_VEC4UC;
        const int format = tex.channels == 3 ? OSP_TEXTURE_RGB8 : OSP_TEXTURE_RGBA8;
        const int filter = OSP_TEXTURE_FILTER_BILINEAR;

        OSPData tex_data = ospNewSharedData(tex.img.data(), data_type, tex.width, 0, tex.height, 0);
        OSPTexture t = ospNewTexture("texture2d");
        ospSetParam(t, "format", OSP_INT, &format);
        ospSetParam(t, "filter", OSP_INT, &filter);
        ospSetParam(t, "data", OSP_DATA, &tex_data);
        ospCommit(t);
        textures.push_back(t);
    }

    materials.clear();
    for (const auto &mat : scene.materials) {
        OSPMaterial m = ospNewMaterial("pathtracer", "Principled");
        ospSetParam(m, "baseColor", OSP_VEC3F, &mat.base_color.x);
        if (mat.color_tex_id != -1) {
            ospSetParam(m, "map_baseColor", OSP_TEXTURE, &textures[mat.color_tex_id]);
        }

        ospSetParam(m, "metallic", OSP_FLOAT, &mat.metallic);
        // TODO: Seems like "specular" here doesn't mean quite what I expect for the Disney BRDF
        // ospSetParam(m, "specular", OSP_FLOAT, &mat.specular);
        ospSetParam(m, "roughness", OSP_FLOAT, &mat.roughness);
        // TODO: name for "specularTint" in OSPRay's model?
        ospSetParam(m, "anisotropy", OSP_FLOAT, &mat.anisotropy);
        ospSetParam(m, "sheen", OSP_FLOAT, &mat.sheen);
        ospSetParam(m, "sheenTint", OSP_FLOAT, &mat.sheen_tint);
        ospSetParam(m, "coat", OSP_FLOAT, &mat.clearcoat);
        // TODO: need to mape clearcoat gloss to ospray
        ospSetParam(m, "ior", OSP_FLOAT, &mat.ior);
        ospSetParam(m, "transmission", OSP_FLOAT, &mat.specular_transmission);

        ospCommit(m);
        materials.push_back(m);
    }

    // TODO: We don't ever have a geometry w/o a material tied to it for now b/c of
    // how the geometry is setup (pairing a mesh w/ a material). This pairing should be split later
    meshes.clear();
    for (const auto &mesh : scene.meshes) {
        std::vector<OSPGeometricModel> geometries;
        for (const auto &geom : mesh.geometries) {
            OSPData verts_data =
                ospNewSharedData(geom.vertices.data(), OSP_VEC3F, geom.vertices.size());
            OSPData indices_data =
                ospNewSharedData(geom.indices.data(), OSP_VEC3UI, geom.indices.size());

            OSPGeometry g = ospNewGeometry("triangles");
            ospSetParam(g, "vertex.position", OSP_DATA, &verts_data);
            ospSetParam(g, "index", OSP_DATA, &indices_data);

            if (!geom.uvs.empty()) {
                OSPData uv_data = ospNewSharedData(geom.uvs.data(), OSP_VEC2F, geom.uvs.size());
                ospSetParam(g, "vertex.texcoord", OSP_DATA, &uv_data);
            }

            ospCommit(g);

            OSPData mat_list = ospNewSharedData(&materials[geom.material_id], OSP_MATERIAL, 1);

            OSPGeometricModel geom_model = ospNewGeometricModel(g);
            ospSetParam(geom_model, "material", OSP_DATA, &mat_list);
            ospCommit(geom_model);
            geometries.push_back(geom_model);
            ospRelease(g);
        }
        OSPData stage_geoms_data =
            ospNewSharedData(geometries.data(), OSP_GEOMETRIC_MODEL, geometries.size());
        OSPData geoms_data = ospNewData(OSP_GEOMETRIC_MODEL, geometries.size());
        ospCopyData(stage_geoms_data, geoms_data);

        OSPGroup group = ospNewGroup();
        ospSetParam(group, "geometry", OSP_DATA, &geoms_data);
        ospCommit(group);

        meshes.push_back(group);
    }

    instances.clear();
    for (const auto &inst : scene.instances) {
        OSPInstance osp_instance = ospNewInstance(meshes[inst.mesh_id]);
        const glm::mat4x3 m(inst.transform);
        ospSetParam(osp_instance, "xfm", OSP_AFFINE3F, glm::value_ptr(m));
        ospCommit(osp_instance);
        instances.push_back(osp_instance);
    }

    lights.clear();
    for (const auto &light : scene.lights) {
        OSPLight l = ospNewLight("quad");

        const glm::vec3 color = glm::normalize(glm::vec3(light.emission));
        const float intensity = glm::length(glm::vec3(light.emission));
        const bool visible = false;
        ospSetParam(l, "color", OSP_VEC3F, &color.x);
        ospSetParam(l, "intensity", OSP_FLOAT, &intensity);
        ospSetParam(l, "visible", OSP_BOOL, &visible);

        const glm::vec3 edgex = light.v_x * light.width;
        const glm::vec3 edgey = light.v_y * light.height;
        ospSetParam(l, "position", OSP_VEC3F, &light.position);
        ospSetParam(l, "edge1", OSP_VEC3F, &edgex.x);
        ospSetParam(l, "edge2", OSP_VEC3F, &edgey.x);
        ospCommit(l);
        lights.push_back(l);
    }

    {
        OSPLight ambient = ospNewLight("ambient");
        float ambient_intensity = 0.2f;
        ospSetParam(ambient, "intensity", OSP_FLOAT, &ambient_intensity);
        ospCommit(ambient);
        lights.push_back(ambient);
    }

    OSPData instances_list = ospNewSharedData(instances.data(), OSP_INSTANCE, instances.size());
    OSPData lights_list = ospNewSharedData(lights.data(), OSP_LIGHT, lights.size());

    world = ospNewWorld();
    ospSetParam(world, "instance", OSP_DATA, &instances_list);
    ospSetParam(world, "light", OSP_DATA, &lights_list);
    ospCommit(world);
}

RenderStats RenderOSPRay::render(const glm::vec3 &pos,
                                 const glm::vec3 &dir,
                                 const glm::vec3 &up,
                                 const float fovy,
                                 const bool camera_changed)
{
    using namespace std::chrono;
    if (camera_changed) {
        ospSetParam(camera, "position", OSP_VEC3F, &pos.x);
        ospSetParam(camera, "direction", OSP_VEC3F, &dir.x);
        ospSetParam(camera, "up", OSP_VEC3F, &up.x);
        ospSetParam(camera, "fovy", OSP_FLOAT, &fovy);
        ospCommit(camera);
        ospResetAccumulation(fb);
    }

    RenderStats stats;
    auto start = high_resolution_clock::now();
    OSPFuture future = ospRenderFrame(fb, renderer, camera, world);
    ospWait(future);
    auto end = high_resolution_clock::now();
    stats.render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-6;

    const uint32_t *mapped = static_cast<const uint32_t *>(ospMapFrameBuffer(fb, OSP_FB_COLOR));
    std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
    ospUnmapFrameBuffer(mapped, fb);

    return stats;
}

