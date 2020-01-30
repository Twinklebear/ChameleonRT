#include "render_ospray.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <tbb/parallel_for.h>
#include "texture_channel_mask.h"
#include "util.h"
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
    // Apply a y-flip to the image to match the other backends which render
    // in the DirectX/Vulkan image coordinate system
    const glm::vec2 img_start(0.f, 1.f);
    const glm::vec2 img_end(1.f, 0.f);
    ospSetParam(camera, "imageStart", OSP_VEC2F, &img_start.x);
    ospSetParam(camera, "imageEnd", OSP_VEC2F, &img_end.x);

    renderer = ospNewRenderer("pathtracer");
    ospCommit(renderer);

    world = ospNewWorld();
}

RenderOSPRay::~RenderOSPRay()
{
    for (auto &t : textures) {
        ospRelease(t);
    }
    for (auto &m : materials) {
        ospRelease(m);
    }
    for (auto &i : instances) {
        ospRelease(i);
    }
    for (auto &l : lights) {
        ospRelease(l);
    }
    ospRelease(camera);
    ospRelease(renderer);
    ospRelease(fb);
    ospRelease(world);
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

    // Linearize any sRGB textures beforehand, since we don't have fancy sRGB texture
    // interpolation support in hardware
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

    for (auto &t : textures) {
        ospRelease(t);
    }
    textures.clear();
    for (const auto &tex : scene.textures) {
        const OSPDataType data_type = tex.channels == 3 ? OSP_VEC3UC : OSP_VEC4UC;
        const int format = tex.channels == 3 ? OSP_TEXTURE_RGB8 : OSP_TEXTURE_RGBA8;
        const int filter = OSP_TEXTURE_FILTER_BILINEAR;

        OSPData tex_data =
            ospNewSharedData(tex.img.data(), data_type, tex.width, 0, tex.height, 0);
        OSPTexture t = ospNewTexture("texture2d");
        ospSetParam(t, "format", OSP_INT, &format);
        ospSetParam(t, "filter", OSP_INT, &filter);
        ospSetParam(t, "data", OSP_DATA, &tex_data);
        ospCommit(t);
        textures.push_back(t);
    }

    for (auto &m : materials) {
        ospRelease(m);
    }
    materials.clear();
    for (const auto &mat : scene.materials) {
        OSPMaterial m = ospNewMaterial("pathtracer", "principled");
        const int tex_handle = *reinterpret_cast<const int *>(&mat.base_color.x);
        if (IS_TEXTURED_PARAM(tex_handle)) {
            ospSetParam(
                m, "map_baseColor", OSP_TEXTURE, &textures[GET_TEXTURE_ID(tex_handle)]);
        } else {
            ospSetParam(m, "baseColor", OSP_VEC3F, &mat.base_color.x);
        }

        set_material_param(m, "metallic", mat.metallic);
        // TODO: Seems like "specular" here means something really different and weird or is
        // buggy
        // set_material_param(m, "specular", mat.specular);
        set_material_param(m, "roughness", mat.roughness);
        // TODO: name for "specularTint" in OSPRay's model?
        set_material_param(m, "anisotropy", mat.anisotropy);
        set_material_param(m, "sheen", mat.sheen);
        set_material_param(m, "sheenTint", mat.sheen_tint);
        set_material_param(m, "coat", mat.clearcoat);
        // TODO: need to map clearcoat gloss to ospray
        set_material_param(m, "ior", mat.ior);
        set_material_param(m, "transmission", mat.specular_transmission);

        ospCommit(m);
        materials.push_back(m);
    }
    {
        OSPData material_list =
            ospNewSharedData(materials.data(), OSP_MATERIAL, materials.size());
        ospSetParam(renderer, "material", OSP_DATA, &material_list);
        ospCommit(renderer);
    }

    std::vector<std::vector<OSPGeometry>> meshes;
    for (const auto &mesh : scene.meshes) {
        std::vector<OSPGeometry> mesh_geometries;
        for (const auto &geom : mesh.geometries) {
            OSPData verts_data =
                ospNewSharedData(geom.vertices.data(), OSP_VEC3F, geom.vertices.size());
            OSPData indices_data =
                ospNewSharedData(geom.indices.data(), OSP_VEC3UI, geom.indices.size());

            OSPGeometry g = ospNewGeometry("mesh");
            ospSetParam(g, "vertex.position", OSP_DATA, &verts_data);
            ospSetParam(g, "index", OSP_DATA, &indices_data);

            if (!geom.uvs.empty()) {
                OSPData uv_data =
                    ospNewSharedData(geom.uvs.data(), OSP_VEC2F, geom.uvs.size());
                ospSetParam(g, "vertex.texcoord", OSP_DATA, &uv_data);
            }
            ospCommit(g);
            mesh_geometries.push_back(g);
        }
        meshes.push_back(mesh_geometries);
    }

    for (auto &i : instances) {
        ospRelease(i);
    }
    instances.clear();
    for (const auto &inst : scene.instances) {
        // Make models for each geometry in the instance's mesh to set the material
        std::vector<OSPGeometricModel> geom_models;
        for (size_t i = 0; i < meshes[inst.mesh_id].size(); ++i) {
            OSPGeometricModel gm = ospNewGeometricModel(meshes[inst.mesh_id][i]);
            ospSetParam(gm, "material", OSP_UINT, &inst.material_ids[i]);
            ospCommit(gm);
            geom_models.push_back(gm);
        }

        OSPGroup group = ospNewGroup();
        OSPData stage =
            ospNewSharedData(geom_models.data(), OSP_GEOMETRIC_MODEL, geom_models.size());
        OSPData geom_list = ospNewData(OSP_GEOMETRIC_MODEL, geom_models.size());
        ospCopyData(stage, geom_list);
        ospSetParam(group, "geometry", OSP_DATA, &geom_list);
        ospCommit(group);

        OSPInstance osp_instance = ospNewInstance(group);
        const glm::mat4x3 m(inst.transform);
        ospSetParam(osp_instance, "xfm", OSP_AFFINE3F, glm::value_ptr(m));
        ospCommit(osp_instance);
        instances.push_back(osp_instance);

        // Ref-counted internally by OSPRay, we no longer need these handles
        for (auto &gm : geom_models) {
            ospRelease(gm);
        }
        ospRelease(group);
    }
    // Ref-counted internally by OSPRay, we no longer need these handles
    for (auto &m : meshes) {
        for (auto &g : m) {
            ospRelease(g);
        }
    }

    for (auto &l : lights) {
        ospRelease(l);
    }
    lights.clear();
    for (const auto &light : scene.lights) {
        OSPLight l = ospNewLight("quad");

        const glm::vec3 color = glm::normalize(glm::vec3(light.emission));
        const float intensity = glm::length(glm::vec3(light.emission));
        const bool visible = true;
        ospSetParam(l, "color", OSP_VEC3F, &color.x);
        ospSetParam(l, "intensity", OSP_FLOAT, &intensity);
        ospSetParam(l, "visible", OSP_BOOL, &visible);

        glm::vec3 edgex = light.v_x * light.width;
        glm::vec3 edgey = light.v_y * light.height;
        // Make sure the light is oriented facing the direction we want. OSPRay determines this
        // by the cross produce of the two edges
        if (glm::dot(glm::cross(edgex, edgey), glm::vec3(light.normal)) < 0.f) {
            std::swap(edgex, edgey);
        }
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

    OSPData instances_list =
        ospNewSharedData(instances.data(), OSP_INSTANCE, instances.size());
    OSPData lights_list = ospNewSharedData(lights.data(), OSP_LIGHT, lights.size());

    ospSetParam(world, "instance", OSP_DATA, &instances_list);
    ospSetParam(world, "light", OSP_DATA, &lights_list);
    ospCommit(world);
}

RenderStats RenderOSPRay::render(const glm::vec3 &pos,
                                 const glm::vec3 &dir,
                                 const glm::vec3 &up,
                                 const float fovy,
                                 const bool camera_changed,
                                 const bool need_readback)
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

    const uint32_t *mapped =
        static_cast<const uint32_t *>(ospMapFrameBuffer(fb, OSP_FB_COLOR));
    std::memcpy(img.data(), mapped, sizeof(uint32_t) * img.size());
    ospUnmapFrameBuffer(mapped, fb);

    return stats;
}

void RenderOSPRay::set_material_param(OSPMaterial &mat,
                                      const std::string &name,
                                      const float val) const
{
    const uint32_t handle = *reinterpret_cast<const uint32_t *>(&val);
    if (IS_TEXTURED_PARAM(handle)) {
        const std::string map_name = "map_" + name;
        ospSetParam(mat, map_name.c_str(), OSP_TEXTURE, &textures[GET_TEXTURE_ID(handle)]);
    } else {
        ospSetParam(mat, name.c_str(), OSP_FLOAT, &val);
    }
}

