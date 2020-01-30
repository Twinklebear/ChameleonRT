#include "chameleon_device.h"
#include <iostream>

using namespace ospcommon::math;

int ChameleonDevice::loadModule(const char *name)
{
    std::cout << "ChameleonDevice cannot be used with other modules\n";
    return OSP_INVALID_OPERATION;
}

OSPData ChameleonDevice::newSharedData(const void *sharedData,
                                       OSPDataType,
                                       const vec3ul &numItems,
                                       const vec3l &byteStride)
{
    return 0;
}

OSPData ChameleonDevice::newData(OSPDataType, const vec3ul &numItems)
{
    return 0;
}

void ChameleonDevice::copyData(const OSPData source,
                               OSPData destination,
                               const vec3ul &destinationIndex)
{
}

OSPLight ChameleonDevice::newLight(const char *type)
{
    return 0;
}

OSPCamera ChameleonDevice::newCamera(const char *type)
{
    return 0;
}

OSPGeometry ChameleonDevice::newGeometry(const char *type)
{
    return 0;
}
OSPVolume ChameleonDevice::newVolume(const char *type)
{
    return 0;
}

OSPGeometricModel ChameleonDevice::newGeometricModel(OSPGeometry geom)
{
    return 0;
}

OSPVolumetricModel ChameleonDevice::newVolumetricModel(OSPVolume volume)
{
    return 0;
}

OSPMaterial ChameleonDevice::newMaterial(const char *renderer_type, const char *material_type)
{
    return 0;
}

OSPTransferFunction ChameleonDevice::newTransferFunction(const char *type)
{
    return 0;
}

OSPTexture ChameleonDevice::newTexture(const char *type)
{
    return 0;
}

OSPGroup ChameleonDevice::newGroup()
{
    return 0;
}

OSPInstance ChameleonDevice::newInstance(OSPGroup group)
{
    return 0;
}

OSPWorld ChameleonDevice::newWorld()
{
    return 0;
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

void ChameleonDevice::release(OSPObject _obj) {}

void ChameleonDevice::retain(OSPObject _obj) {}

OSPFrameBuffer ChameleonDevice::frameBufferCreate(const vec2i &size,
                                                  const OSPFrameBufferFormat mode,
                                                  const uint32_t channels)
{
    return 0;
}

OSPImageOperation ChameleonDevice::newImageOp(const char *type)
{
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

extern "C" OSPError OSPRAY_MODULE_CHAMELEON_EXPORT
ospray_module_init_chameleon(int16_t versionMajor, int16_t versionMinor, int16_t versionPatch)
{
    std::cout << "ChameleonRT module loaded\n";
    return ospray::moduleVersionCheck(versionMajor, versionMinor);
}
