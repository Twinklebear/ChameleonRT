#include "chameleon_device.h"
#include <iostream>
#include <stdexcept>
#include <ospcommon/utility/ParameterizedObject.h>

using namespace ospcommon::math;

namespace device {

int ChameleonDevice::loadModule(const char *name)
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
    Data *src = dynamic_cast<Data *>(reinterpret_cast<APIObject *>(source));
    OwnedData *dst = dynamic_cast<OwnedData *>(reinterpret_cast<APIObject *>(dst));
    if (!src) {
        throw std::runtime_error("Copy source must be a Data");
    }
    if (!dst) {
        throw std::runtime_error("Copy destination must be a device-owned Data (ospNewData)");
    }

    std::memcpy(dst->data(), src->data(), src->size_bytes());
}

OSPLight ChameleonDevice::newLight(const char *type)
{
    Light *obj = new Light();
    return reinterpret_cast<OSPLight>(obj);
}

OSPCamera ChameleonDevice::newCamera(const char *type)
{
    Camera *obj = new Camera();
    return reinterpret_cast<OSPCamera>(obj);
}

OSPGeometry ChameleonDevice::newGeometry(const char *type)
{
    Geometry *obj = new Geometry();
    return reinterpret_cast<OSPGeometry>(obj);
}

OSPVolume ChameleonDevice::newVolume(const char *type)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPGeometricModel ChameleonDevice::newGeometricModel(OSPGeometry geom)
{
    GeometricModel *obj = new GeometricModel(reinterpret_cast<Geometry *>(geom));
    return reinterpret_cast<OSPGeometricModel>(obj);
}

OSPVolumetricModel ChameleonDevice::newVolumetricModel(OSPVolume volume)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPMaterial ChameleonDevice::newMaterial(const char *renderer_type, const char *material_type)
{
    Material *obj = new Material();
    return reinterpret_cast<OSPMaterial>(obj);
}

OSPTransferFunction ChameleonDevice::newTransferFunction(const char *type)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPTexture ChameleonDevice::newTexture(const char *type)
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
        const Data *data = reinterpret_cast<const Data *>(handle);
        object->setParam(name, data);
        break;
    }
    case OSP_TEXTURE: {
        const Texture *tex = reinterpret_cast<const Texture *>(handle);
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

void ChameleonDevice::release(OSPObject obj)
{
    // Note: Intentionally does nothing, since we do need a manual ref counted pointer
    // and extra work in the Data implementation for the retain/release and internal
    // ref count tracking to work
}

void ChameleonDevice::retain(OSPObject _obj) {}

OSPFrameBuffer ChameleonDevice::frameBufferCreate(const vec2i &size,
                                                  const OSPFrameBufferFormat mode,
                                                  const uint32_t channels)
{
    Framebuffer *obj = new Framebuffer(size);
    return reinterpret_cast<OSPFrameBuffer>(obj);
}

OSPImageOperation ChameleonDevice::newImageOp(const char *type)
{
    throw std::runtime_error("ImageOps are not supported by ChameleonRT");
    return 0;
}

const void *ChameleonDevice::frameBufferMap(OSPFrameBuffer fb, const OSPFrameBufferChannel)
{
    return nullptr;
}

void ChameleonDevice::frameBufferUnmap(const void *mapped, OSPFrameBuffer fb) {}

float ChameleonDevice::getVariance(OSPFrameBuffer)
{
    return 0.f;
}

void ChameleonDevice::resetAccumulation(OSPFrameBuffer _fb) {}

OSPRenderer ChameleonDevice::newRenderer(const char *type)
{
    Renderer *obj = new Renderer();
    return reinterpret_cast<OSPRenderer>(obj);
}

OSPFuture ChameleonDevice::renderFrame(OSPFrameBuffer fb_handle,
                                       OSPRenderer renderer_handle,
                                       OSPCamera camera_handle,
                                       OSPWorld world_handle)
{
    return 0;
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

void ChameleonDevice::commit() {}

extern "C" OSPError OSPRAY_MODULE_CHAMELEON_EXPORT ospray_module_init_chameleon(
    int16_t version_major, int16_t version_minor, int16_t version_patch)
{
    std::cout << "ChameleonRT module loaded\n";
    return ospray::moduleVersionCheck(version_major, version_minor);
}
}
