#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <ospcommon/math/box.h>
#include <ospcommon/math/vec.h>
#include <ospray/OSPEnums.h>
#include <ospray/SDK/api/Device.h>
#include <ospray/SDK/common/OSPCommon.h>
#include "data.h"
#include "mesh.h"
#include "objects.h"
#include "ospray_module_chameleon_export.h"
#include "scene.h"

namespace device {

// Note: Device just uses raw pointers throughout and doesn't implement
// the ref-counting and lifetime management parts of the API
struct OSPRAY_MODULE_CHAMELEON_EXPORT ChameleonDevice : public ospray::api::Device {
    ChameleonDevice() = default;
    ~ChameleonDevice() override = default;

    /////////////////////////////////////////////////////////////////////////
    // Main interface to accepting API calls
    /////////////////////////////////////////////////////////////////////////

    // Modules //////////////////////////////////////////////////////////////

    int loadModule(const char *name) override;

    // OSPRay Data Arrays ///////////////////////////////////////////////////

    OSPData newSharedData(const void *shared_data,
                          OSPDataType type,
                          const ospcommon::math::vec3ul &num_items,
                          const ospcommon::math::vec3l &byte_stride) override;

    OSPData newData(OSPDataType type, const ospcommon::math::vec3ul &num_items) override;

    void copyData(const OSPData source,
                  OSPData destination,
                  const ospcommon::math::vec3ul &dest_index) override;

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

    ospcommon::math::box3f getBounds(OSPObject object) override;

    // Object + Parameter Lifetime Management ///////////////////////////////

    void setObjectParam(OSPObject object,
                        const char *name,
                        OSPDataType type,
                        const void *mem) override;

    void removeObjectParam(OSPObject object, const char *name) override;

    void commit(OSPObject object) override;

    void release(OSPObject object) override;

    void retain(OSPObject object) override;

    // FrameBuffer Manipulation /////////////////////////////////////////////

    OSPFrameBuffer frameBufferCreate(const ospcommon::math::vec2i &size,
                                     const OSPFrameBufferFormat mode,
                                     const uint32_t channels) override;

    OSPImageOperation newImageOp(const char *type) override;

    const void *frameBufferMap(OSPFrameBuffer fb,
                               const OSPFrameBufferChannel channel) override;

    void frameBufferUnmap(const void *mapped, OSPFrameBuffer fb) override;

    float getVariance(OSPFrameBuffer fb) override;

    void resetAccumulation(OSPFrameBuffer fb) override;

    // Frame Rendering //////////////////////////////////////////////////////

    OSPRenderer newRenderer(const char *type) override;

    OSPFuture renderFrame(OSPFrameBuffer fb,
                          OSPRenderer renderer,
                          OSPCamera camera,
                          OSPWorld world) override;

    int isReady(OSPFuture future, OSPSyncEvent event) override;

    void wait(OSPFuture future, OSPSyncEvent event) override;

    void cancel(OSPFuture future) override;

    float getProgress(OSPFuture future) override;

    /////////////////////////////////////////////////////////////////////////
    // Helper/other functions and data members
    /////////////////////////////////////////////////////////////////////////

    void commit() override;
};

OSP_REGISTER_DEVICE(ChameleonDevice, chameleon);
}
