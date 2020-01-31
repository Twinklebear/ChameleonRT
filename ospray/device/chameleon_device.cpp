#include "chameleon_device.h"
#include <iostream>
#include <stdexcept>
#include <ospcommon/utility/ParameterizedObject.h>

#if ENABLE_OPTIX
#include "optix/render_optix.h"
#endif
#if ENABLE_EMBREE
#include "embree/render_embree.h"
#endif
#if ENABLE_DXR
#include "dxr/render_dxr.h"
#endif
#if ENABLE_VULKAN
#include "vulkan/render_vulkan.h"
#endif

using namespace ospcommon::math;

namespace device {

int ChameleonDevice::loadModule(const char *)
{
    std::cout << "ChameleonDevice cannot be used with other modules\n";
    return OSP_INVALID_OPERATION;
}

OSPData ChameleonDevice::newSharedData(const void *shared_data,
                                       OSPDataType type,
                                       const vec3ul &num_items,
                                       const vec3l &byte_stride)
{
    if (byte_stride != vec3l(0) && byte_stride.x != ospray::sizeOf(type) &&
        byte_stride.y != byte_stride.x * num_items.x &&
        byte_stride.z != num_items.y * byte_stride.y) {
        throw std::runtime_error("ChameleonRT device only supports compact data");
    }
    BorrowedData *data =
        new BorrowedData(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(shared_data)),
                         num_items,
                         type);
    return reinterpret_cast<OSPData>(data);
}

OSPData ChameleonDevice::newData(OSPDataType type, const vec3ul &num_items)
{
    OwnedData *data = new OwnedData(num_items, type);
    return reinterpret_cast<OSPData>(data);
}

void ChameleonDevice::copyData(const OSPData source,
                               OSPData destination,
                               const vec3ul &dest_index)
{
    if (dest_index != vec3ul(0)) {
        throw std::runtime_error("ChameleonRT device only supports memcpy copies");
    }
    Data *src = reinterpret_cast<Data *>(source);
    OwnedData *dst = reinterpret_cast<OwnedData *>(destination);
    std::memcpy(dst->data(), src->data(), src->size_bytes());
}

OSPLight ChameleonDevice::newLight(const char *)
{
    Light *obj = new Light();
    return reinterpret_cast<OSPLight>(obj);
}

OSPCamera ChameleonDevice::newCamera(const char *)
{
    Camera *obj = new Camera();
    return reinterpret_cast<OSPCamera>(obj);
}

OSPGeometry ChameleonDevice::newGeometry(const char *)
{
    Geometry *obj = new Geometry();
    return reinterpret_cast<OSPGeometry>(obj);
}

OSPVolume ChameleonDevice::newVolume(const char *)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPGeometricModel ChameleonDevice::newGeometricModel(OSPGeometry geom)
{
    GeometricModel *obj = new GeometricModel(reinterpret_cast<Geometry *>(geom));
    return reinterpret_cast<OSPGeometricModel>(obj);
}

OSPVolumetricModel ChameleonDevice::newVolumetricModel(OSPVolume)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPMaterial ChameleonDevice::newMaterial(const char *, const char *)
{
    Material *obj = new Material();
    return reinterpret_cast<OSPMaterial>(obj);
}

OSPTransferFunction ChameleonDevice::newTransferFunction(const char *)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPTexture ChameleonDevice::newTexture(const char *)
{
    Texture *obj = new Texture();
    return reinterpret_cast<OSPTexture>(obj);
}

OSPGroup ChameleonDevice::newGroup()
{
    Group *obj = new Group();
    return reinterpret_cast<OSPGroup>(obj);
}

OSPInstance ChameleonDevice::newInstance(OSPGroup group)
{
    Instance *obj = new Instance(reinterpret_cast<Group *>(group));
    return reinterpret_cast<OSPInstance>(obj);
}

OSPWorld ChameleonDevice::newWorld()
{
    World *obj = new World();
    return reinterpret_cast<OSPWorld>(obj);
}

box3f ChameleonDevice::getBounds(OSPObject)
{
    throw std::runtime_error("Chameleon Device does not support bounds queries");
    return box3f{};
}

void ChameleonDevice::setObjectParam(OSPObject handle,
                                     const char *name,
                                     OSPDataType type,
                                     const void *mem)
{
    // Just the params I need for the OSPRay backend in ChameleonRT
    APIObject *object = reinterpret_cast<APIObject *>(handle);
    switch (type) {
    case OSP_BOOL:
        object->setParam(name, *reinterpret_cast<const bool *>(mem));
        break;
    case OSP_FLOAT:
        object->setParam(name, *reinterpret_cast<const float *>(mem));
        break;
    case OSP_INT:
        object->setParam(name, *reinterpret_cast<const int *>(mem));
        break;
    case OSP_UINT:
        object->setParam(name, *reinterpret_cast<const uint32_t *>(mem));
        break;
    case OSP_VEC2F:
        object->setParam(name, *reinterpret_cast<const vec2f *>(mem));
        break;
    case OSP_VEC3F:
        object->setParam(name, *reinterpret_cast<const vec3f *>(mem));
        break;
    case OSP_AFFINE3F:
        object->setParam(name, *reinterpret_cast<const affine3f *>(mem));
        break;
    case OSP_DATA: {
        Data *data = *(Data **)(mem);
        object->setParam(name, data);
        break;
    }
    case OSP_TEXTURE: {
        Texture *tex = *(Texture **)(mem);
        object->setParam(name, tex);
        break;
    }
    default:
        throw std::runtime_error("Parameter " + ospray::stringFor(type) +
                                 " is not handled by ChameleonRT");
    }
}

void ChameleonDevice::removeObjectParam(OSPObject handle, const char *name)
{
    // Just the params I need for the OSPRay backend in ChameleonRT
    APIObject *object = reinterpret_cast<APIObject *>(handle);
    object->removeParam(name);
}

void ChameleonDevice::commit(OSPObject handle)
{
    // We only really care about committing the scene in this kind of hack device
    APIObject *obj = reinterpret_cast<APIObject *>(handle);
    obj->commit();
}

void ChameleonDevice::release(OSPObject)
{
    // Note: Intentionally does nothing, since we do need a manual ref counted pointer
    // and extra work in the Data implementation for the retain/release and internal
    // ref count tracking to work
}

void ChameleonDevice::retain(OSPObject) {}

OSPFrameBuffer ChameleonDevice::frameBufferCreate(const vec2i &size,
                                                  const OSPFrameBufferFormat,
                                                  const uint32_t)
{
    Framebuffer *obj = new Framebuffer(size);
    return reinterpret_cast<OSPFrameBuffer>(obj);
}

OSPImageOperation ChameleonDevice::newImageOp(const char *)
{
    throw std::runtime_error("ImageOps are not supported by ChameleonRT");
    return 0;
}

const void *ChameleonDevice::frameBufferMap(OSPFrameBuffer handle, const OSPFrameBufferChannel)
{
    Framebuffer *fb = reinterpret_cast<Framebuffer *>(handle);
    return fb->img.data();
}

void ChameleonDevice::frameBufferUnmap(const void *, OSPFrameBuffer) {}

float ChameleonDevice::getVariance(OSPFrameBuffer)
{
    return 0.f;
}

void ChameleonDevice::resetAccumulation(OSPFrameBuffer handle)
{
    Framebuffer *fb = reinterpret_cast<Framebuffer *>(handle);
    fb->accum_id = 0;
}

OSPRenderer ChameleonDevice::newRenderer(const char *)
{
    Renderer *obj = new Renderer();
    return reinterpret_cast<OSPRenderer>(obj);
}

OSPFuture ChameleonDevice::renderFrame(OSPFrameBuffer fb_handle,
                                       OSPRenderer renderer_handle,
                                       OSPCamera camera_handle,
                                       OSPWorld world_handle)
{
    Framebuffer *fb = reinterpret_cast<Framebuffer *>(fb_handle);
    Renderer *renderer = reinterpret_cast<Renderer *>(renderer_handle);
    Camera *camera = reinterpret_cast<Camera *>(camera_handle);
    World *world = reinterpret_cast<World *>(world_handle);

    if (renderer->last_framebuffer != fb) {
        render_backend->initialize(fb->size.x, fb->size.y);
        renderer->last_framebuffer = fb;
    }
    if (renderer->last_world != world) {
        world->scene.materials = renderer->materials;
        world->scene.textures = renderer->images;
        render_backend->set_scene(world->scene);
        renderer->last_world = world;
    }

    render_backend->render(camera->camera.position,
                           glm::normalize(camera->camera.center - camera->camera.position),
                           camera->camera.up,
                           camera->camera.fov_y,
                           fb->accum_id == 0,
                           true);

    std::memcpy(fb->img.data(), render_backend->img.data(), sizeof(uint32_t) * fb->img.size());

    fb->accum_id++;

    return reinterpret_cast<OSPFuture>(1);
}

int ChameleonDevice::isReady(OSPFuture, OSPSyncEvent)
{
    return 1;
}

void ChameleonDevice::wait(OSPFuture, OSPSyncEvent) {}

void ChameleonDevice::cancel(OSPFuture) {}

float ChameleonDevice::getProgress(OSPFuture)
{
    return 1.f;
}

void ChameleonDevice::commit()
{
    Device::commit();
    const std::string backend = getParam<std::string>("backend", "embree");
    std::cout << "Selecting backend: " << backend << "\n";

#if ENABLE_OPTIX
    if (backend == "optix") {
        render_backend = std::make_unique<RenderOptiX>(false);
    }
#endif
#if ENABLE_EMBREE
    if (backend == "embree") {
        render_backend = std::make_unique<RenderEmbree>();
    }
#endif
#if ENABLE_DXR
    if (backend == "dxr") {
        render_backend = std::make_unique<RenderDXR>();
    }
#endif
#if ENABLE_VULKAN
    if (backend == "vulkan") {
        render_backend = std::make_unique<RenderVulkan>();
    }
#endif
    if (!render_backend) {
        throw std::runtime_error("Request for unsupported renderer backend " + backend);
    }
}

extern "C" OSPError OSPRAY_MODULE_CHAMELEON_EXPORT
ospray_module_init_chameleon(int16_t version_major, int16_t version_minor, int16_t)
{
    return ospray::moduleVersionCheck(version_major, version_minor);
}
}
