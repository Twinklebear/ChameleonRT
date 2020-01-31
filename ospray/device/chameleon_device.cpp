#include "chameleon_device.h"
#include <iostream>
#include <stdexcept>

// High 4 bits of the handle are used to store the object type
#define DATA_HANDLE uint64_t(0)
#define GEOMETRY_HANDLE uint64_t(1)
#define GEOMETRIC_MODEL_HANDLE uint64_t(2)
#define MESH_HANDLE uint64_t(3)
#define INSTANCE_HANDLE uint64_t(4)
#define MATERIAL_HANDLE uint64_t(5)
#define TEXTURE_HANDLE uint64_t(6)
#define LIGHT_HANDLE uint64_t(7)
#define CAMERA_HANDLE uint64_t(8)
#define WORLD_HANDLE uint64_t(9)

#define HANDLE_TYPE_MASK uint64_t(0xf) << 60
#define DATA_HANDLE_MASK (DATA_HANDLE)
#define GEOMETRY_HANDLE_MASK (GEOMETRY_HANDLE << 60)
#define GEOMETRIC_MODEL_HANDLE_MASK (GEOMETRIC_MODEL_HANDLE << 60)
#define MESH_HANDLE_MASK (MESH_HANDLE << 60)
#define INSTANCE_HANDLE_MASK (INSTANCE_HANDLE << 60)
#define MATERIAL_HANDLE_MASK (MATERIAL_HANDLE << 60)
#define TEXTURE_HANDLE_MASK (TEXTURE_HANDLE << 60)
#define LIGHT_HANDLE_MASK (LIGHT_HANDLE << 60)
#define CAMERA_HANDLE_MASK (CAMERA_HANDLE << 60)
#define WORLD_HANDLE_MASK (WORLD_HANDLE << 60)

using namespace ospcommon::math;

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
    const size_t handle = allocate_handle(OSP_DATA);
    data[handle] = std::make_shared<BorrowedData>(
        const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(shared_data)),
        num_items,
        type);
    return reinterpret_cast<OSPData>(handle);
}

OSPData ChameleonDevice::newData(OSPDataType type, const vec3ul &num_items)
{
    const size_t handle = allocate_handle(OSP_DATA);
    data[handle] = std::make_shared<OwnedData>(num_items, type);
    return reinterpret_cast<OSPData>(handle);
}

void ChameleonDevice::copyData(const OSPData source,
                               OSPData destination,
                               const vec3ul &dest_index)
{
    if (dest_index != vec3ul(0)) {
        throw std::runtime_error("ChameleonRT device only supports memcpy copies");
    }
    auto src = data[reinterpret_cast<size_t>(source)];
    auto dst =
        std::dynamic_pointer_cast<OwnedData>(data[reinterpret_cast<size_t>(destination)]);
    if (!dst) {
        throw std::runtime_error("Copy destination must be a device-owned Data (ospNewData)");
    }

    std::memcpy(dst->data(), src->data(), src->size_bytes());
}

OSPLight ChameleonDevice::newLight(const char *type)
{
    const size_t handle = allocate_handle(OSP_LIGHT);
    lights[handle] = std::make_shared<QuadLight>();
    return reinterpret_cast<OSPLight>(handle);
}

OSPCamera ChameleonDevice::newCamera(const char *type)
{
    const size_t handle = allocate_handle(OSP_CAMERA);
    cameras[handle] = std::make_shared<Camera>();
    return reinterpret_cast<OSPCamera>(handle);
}

OSPGeometry ChameleonDevice::newGeometry(const char *type)
{
    const size_t handle = allocate_handle(OSP_GEOMETRY);
    geometries[handle] = std::make_shared<Geometry>();
    return reinterpret_cast<OSPGeometry>(handle);
}

OSPVolume ChameleonDevice::newVolume(const char *type)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPGeometricModel ChameleonDevice::newGeometricModel(OSPGeometry geom)
{
    const size_t handle = allocate_handle(OSP_GEOMETRIC_MODEL);
    geometric_models[handle] = geometries[handle_value(geom)];
    return reinterpret_cast<OSPGeometricModel>(handle);
}

OSPVolumetricModel ChameleonDevice::newVolumetricModel(OSPVolume volume)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPMaterial ChameleonDevice::newMaterial(const char *renderer_type, const char *material_type)
{
    const size_t handle = allocate_handle(OSP_MATERIAL);
    materials[handle] = std::make_shared<DisneyMaterial>();
    return reinterpret_cast<OSPMaterial>(handle);
}

OSPTransferFunction ChameleonDevice::newTransferFunction(const char *type)
{
    throw std::runtime_error("Volumes are not supported by ChameleonRT");
    return 0;
}

OSPTexture ChameleonDevice::newTexture(const char *type)
{
    const size_t handle = allocate_handle(OSP_TEXTURE);
    textures[handle] = std::make_shared<Image>();
    return reinterpret_cast<OSPTexture>(handle);
}

OSPGroup ChameleonDevice::newGroup()
{
    const size_t handle = allocate_handle(OSP_GROUP);
    meshes[handle] = std::make_shared<Mesh>();
    return reinterpret_cast<OSPGroup>(handle);
}

OSPInstance ChameleonDevice::newInstance(OSPGroup group)
{
    const size_t handle = allocate_handle(OSP_INSTANCE);
    instances[handle] =
        std::make_shared<Instance>(glm::mat4(1), handle_value(group), std::vector<uint32_t>{});
    return reinterpret_cast<OSPInstance>(handle);
}

OSPWorld ChameleonDevice::newWorld()
{
    const size_t handle = allocate_handle(OSP_WORLD);
    scenes[handle] = std::make_shared<Scene>();
    return reinterpret_cast<OSPWorld>(handle);
}

box3f ChameleonDevice::getBounds(OSPObject)
{
    return box3f{};
}

void ChameleonDevice::setObjectParam(OSPObject object,
                                     const char *name,
                                     OSPDataType type,
                                     const void *mem)
{
}

void ChameleonDevice::removeObjectParam(OSPObject object, const char *name) {}

void ChameleonDevice::commit(OSPObject object) {}

void ChameleonDevice::release(OSPObject obj) {}

void ChameleonDevice::retain(OSPObject _obj) {}

OSPFrameBuffer ChameleonDevice::frameBufferCreate(const vec2i &size,
                                                  const OSPFrameBufferFormat mode,
                                                  const uint32_t channels)
{
    return 0;
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
    return 0;
}

OSPFuture ChameleonDevice::renderFrame(OSPFrameBuffer, OSPRenderer, OSPCamera, OSPWorld)
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

size_t ChameleonDevice::allocate_handle(OSPDataType type)
{
    size_t h = next_handle++;
    switch (type) {
    case OSP_DATA:
        return h;
    case OSP_GEOMETRY:
        return GEOMETRY_HANDLE_MASK | h;
    case OSP_GEOMETRIC_MODEL:
        return GEOMETRIC_MODEL_HANDLE_MASK | h;
    case OSP_GROUP:
        return MESH_HANDLE_MASK | h;
    case OSP_INSTANCE:
        return INSTANCE_HANDLE_MASK | h;
    case OSP_MATERIAL:
        return MATERIAL_HANDLE_MASK | h;
    case OSP_TEXTURE:
        return TEXTURE_HANDLE_MASK | h;
    case OSP_LIGHT:
        return LIGHT_HANDLE_MASK | h;
    case OSP_CAMERA:
        return CAMERA_HANDLE_MASK | h;
    case OSP_WORLD:
        return WORLD_HANDLE_MASK | h;
    default:
        throw std::runtime_error("Attempt to allocate handle for unsupported type!");
        return 0;
    }
}

size_t ChameleonDevice::handle_value(OSPObject obj)
{
    return reinterpret_cast<size_t>(obj) & (~HANDLE_TYPE_MASK);
}

OSPDataType handle_type(OSPObject obj)
{
    const size_t type = (reinterpret_cast<size_t>(obj) & HANDLE_TYPE_MASK) >> 60;
    switch (type) {
    case DATA_HANDLE:
        return OSP_DATA;
    case GEOMETRY_HANDLE:
        return OSP_GEOMETRY;
    case GEOMETRIC_MODEL_HANDLE:
        return OSP_GEOMETRIC_MODEL;
    case MESH_HANDLE:
        return OSP_GROUP;
    case INSTANCE_HANDLE:
        return OSP_INSTANCE;
    case MATERIAL_HANDLE:
        return OSP_MATERIAL;
    case TEXTURE_HANDLE:
        return OSP_TEXTURE;
    case LIGHT_HANDLE:
        return OSP_LIGHT;
    case CAMERA_HANDLE:
        return OSP_CAMERA;
    case WORLD_HANDLE:
        return OSP_WORLD;
    default:
        throw std::runtime_error("Unrecognized handle type!");
        return OSP_UNKNOWN;
    }
}

extern "C" OSPError OSPRAY_MODULE_CHAMELEON_EXPORT ospray_module_init_chameleon(
    int16_t version_major, int16_t version_minor, int16_t version_patch)
{
    std::cout << "ChameleonRT module loaded\n";
    return ospray::moduleVersionCheck(version_major, version_minor);
}
