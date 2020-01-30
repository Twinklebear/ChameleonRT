#pragma once

#include <cstdint>
#include <ospcommon/math/box.h>
#include <ospcommon/math/vec.h>
#include <ospray/OSPEnums.h>
#include <ospray/SDK/api/Device.h>
#include <ospray/SDK/common/OSPCommon.h>
#include "ospray_module_chameleon_export.h"

struct OSPRAY_MODULE_CHAMELEON_EXPORT ChameleonDevice : public ospray::api::Device {
    ChameleonDevice() = default;
    ~ChameleonDevice() override = default;

    /////////////////////////////////////////////////////////////////////////
    // Main interface to accepting API calls
    /////////////////////////////////////////////////////////////////////////

    // Modules //////////////////////////////////////////////////////////////

    int loadModule(const char *name) override;

    // OSPRay Data Arrays ///////////////////////////////////////////////////

    OSPData newSharedData(const void *sharedData,
                          OSPDataType,
                          const ospcommon::math::vec3ul &numItems,
                          const ospcommon::math::vec3l &byteStride) override;

    OSPData newData(OSPDataType, const ospcommon::math::vec3ul &numItems) override;

    void copyData(const OSPData source,
                  OSPData destination,
                  const ospcommon::math::vec3ul &destinationIndex) override;

    // Renderable Objects ///////////////////////////////////////////////////

    OSPLight newLight(const char *type) override;

    OSPCamera newCamera(const char *type) override;

    OSPGeometry newGeometry(const char *type) override;
    OSPVolume newVolume(const char *type) override;

    OSPGeometricModel newGeometricModel(OSPGeometry geom) override;

    OSPVolumetricModel newVolumetricModel(OSPVolume volume) override;

    // Model Meta-Data //////////////////////////////////////////////////////

    OSPMaterial newMaterial(const char *renderer_type, const char *material_type) override;

    OSPTransferFunction newTransferFunction(const char *type) override;

    OSPTexture newTexture(const char *type) override;

    // Instancing ///////////////////////////////////////////////////////////

    OSPGroup newGroup() override;

    OSPInstance newInstance(OSPGroup group) override;

    // Top-level Worlds /////////////////////////////////////////////////////

    OSPWorld newWorld() override;

    ospcommon::math::box3f getBounds(OSPObject) override;

    // Object + Parameter Lifetime Management ///////////////////////////////

    void setObjectParam(OSPObject object,
                        const char *name,
                        OSPDataType type,
                        const void *mem) override;

    void removeObjectParam(OSPObject object, const char *name) override;

    void commit(OSPObject object) override;

    void release(OSPObject _obj) override;

    void retain(OSPObject _obj) override;

    // FrameBuffer Manipulation /////////////////////////////////////////////

    OSPFrameBuffer frameBufferCreate(const ospcommon::math::vec2i &size,
                                     const OSPFrameBufferFormat mode,
                                     const uint32_t channels) override;

    OSPImageOperation newImageOp(const char *type) override;

    const void *frameBufferMap(OSPFrameBuffer fb, const OSPFrameBufferChannel) override;

    void frameBufferUnmap(const void *mapped, OSPFrameBuffer fb) override;

    float getVariance(OSPFrameBuffer) override;

    void resetAccumulation(OSPFrameBuffer _fb) override;

    // Frame Rendering //////////////////////////////////////////////////////

    OSPRenderer newRenderer(const char *type) override;

    OSPFuture renderFrame(OSPFrameBuffer, OSPRenderer, OSPCamera, OSPWorld) override;

    int isReady(OSPFuture, OSPSyncEvent) override;

    void wait(OSPFuture, OSPSyncEvent) override;

    void cancel(OSPFuture) override;

    float getProgress(OSPFuture) override;

    /////////////////////////////////////////////////////////////////////////
    // Helper/other functions and data members
    /////////////////////////////////////////////////////////////////////////

    void commit() override;
};

OSP_REGISTER_DEVICE(ChameleonDevice, chameleon);
