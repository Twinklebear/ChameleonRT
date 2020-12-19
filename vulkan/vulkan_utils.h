#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#define CHECK_VULKAN(FN)                                   \
    {                                                      \
        VkResult r = FN;                                   \
        if (r != VK_SUCCESS) {                             \
            std::cout << #FN << " failed\n" << std::flush; \
            throw std::runtime_error(#FN " failed!");      \
        }                                                  \
    }

namespace vkrt {

extern PFN_vkCmdTraceRaysKHR CmdTraceRaysKHR;
extern PFN_vkDestroyAccelerationStructureKHR DestroyAccelerationStructureKHR;
extern PFN_vkGetRayTracingShaderGroupHandlesKHR GetRayTracingShaderGroupHandlesKHR;
extern PFN_vkCmdWriteAccelerationStructuresPropertiesKHR
    CmdWriteAccelerationStructuresPropertiesKHR;
extern PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR;
extern PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR;
extern PFN_vkCmdCopyAccelerationStructureKHR CmdCopyAccelerationStructureKHR;
extern PFN_vkCreateRayTracingPipelinesKHR CreateRayTracingPipelinesKHR;
extern PFN_vkGetAccelerationStructureDeviceAddressKHR GetAccelerationStructureDeviceAddressKHR;
extern PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR;

class Device {
    VkInstance vk_instance = VK_NULL_HANDLE;
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;

    uint32_t graphics_queue_index = -1;

    VkPhysicalDeviceMemoryProperties mem_props = {};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props = {};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props = {};

public:
    Device(const std::vector<std::string> &instance_extensions = std::vector<std::string>{},
           const std::vector<std::string> &logical_device_extensions =
               std::vector<std::string>{});
    ~Device();

    Device(Device &&d);
    Device &operator=(Device &&d);

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    VkDevice logical_device();

    VkPhysicalDevice physical_device();

    VkInstance instance();

    VkQueue graphics_queue();
    uint32_t queue_index() const;

    VkCommandPool make_command_pool(
        VkCommandPoolCreateFlagBits flags = (VkCommandPoolCreateFlagBits)0);

    uint32_t memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    VkDeviceMemory alloc(size_t nbytes, uint32_t type_filter, VkMemoryPropertyFlags props);

    const VkPhysicalDeviceMemoryProperties &memory_properties() const;
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR &
    acceleration_structure_properties() const;
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &raytracing_pipeline_properties()
        const;

private:
    void make_instance(const std::vector<std::string> &extensions);
    void select_physical_device();
    void make_logical_device(const std::vector<std::string> &extensions);
};

// TODO: Maybe a base resource class which tracks the queue and access flags

class Buffer {
    size_t buf_size = 0;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    Device *vkdevice = nullptr;
    bool host_visible = false;

    static VkBufferCreateInfo create_info(size_t nbytes, VkBufferUsageFlags usage);

    static std::shared_ptr<Buffer> make_buffer(Device &device,
                                               size_t nbytes,
                                               VkBufferUsageFlags usage,
                                               VkMemoryPropertyFlags mem_props);

public:
    Buffer() = default;
    ~Buffer();
    Buffer(Buffer &&b);
    Buffer &operator=(Buffer &&b);

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    static std::shared_ptr<Buffer> host(
        Device &device,
        size_t nbytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlagBits extra_mem_props = (VkMemoryPropertyFlagBits)0);
    static std::shared_ptr<Buffer> device(
        Device &device,
        size_t nbytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlagBits extra_mem_props = (VkMemoryPropertyFlagBits)0);

    // Map the entire range of the buffer
    void *map();
    // Map a subset of the buffer starting at offset of some size
    void *map(size_t offset, size_t size);

    void unmap();

    size_t size() const;

    VkBuffer handle() const;

    VkDeviceAddress device_address() const;
};

class Texture2D {
    glm::uvec2 tdims = glm::uvec2(0);
    VkFormat img_format;
    VkImageLayout img_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    Device *vkdevice = nullptr;

public:
    Texture2D() = default;
    ~Texture2D();
    Texture2D(Texture2D &&t);
    Texture2D &operator=(Texture2D &&t);

    Texture2D(Texture2D &t) = delete;
    Texture2D &operator=(Texture2D &t) = delete;

    // Note after creation image will be in the image_layout_undefined layout
    static std::shared_ptr<Texture2D> device(Device &device,
                                             glm::uvec2 dims,
                                             VkFormat img_format,
                                             VkImageUsageFlags usage);

    // Size of one pixel, in bytes
    size_t pixel_size() const;
    VkFormat pixel_format() const;
    glm::uvec2 dims() const;

    VkImage image_handle() const;
    VkImageView view_handle() const;
};

struct ShaderModule {
    Device *device = nullptr;
    VkShaderModule module = VK_NULL_HANDLE;

    ShaderModule() = default;
    ShaderModule(Device &device, const uint32_t *code, size_t code_size);
    ~ShaderModule();

    ShaderModule(ShaderModule &&sm);
    ShaderModule &operator=(ShaderModule &&sm);

    ShaderModule(ShaderModule &) = delete;
    ShaderModule &operator=(ShaderModule &) = delete;
};

class DescriptorSetLayoutBuilder {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlagsEXT> binding_ext_flags;

public:
    DescriptorSetLayoutBuilder &add_binding(uint32_t binding,
                                            uint32_t count,
                                            VkDescriptorType type,
                                            uint32_t stage_flags,
                                            uint32_t ext_flags = 0);

    VkDescriptorSetLayout build(Device &device);
};

class TopLevelBVH;

struct WriteDescriptorInfo {
    VkDescriptorSet dst_set = VK_NULL_HANDLE;
    uint32_t binding = 0;
    uint32_t count = 0;
    VkDescriptorType type;
    size_t as_index = -1;
    size_t img_index = -1;
    size_t buf_index = -1;
};

struct CombinedImageSampler {
    const std::shared_ptr<Texture2D> texture;
    VkSampler sampler;

    CombinedImageSampler(const std::shared_ptr<Texture2D> &t, VkSampler sampler);
};

class DescriptorSetUpdater {
    std::vector<WriteDescriptorInfo> writes;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accel_structs;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<VkDescriptorBufferInfo> buffers;

public:
    DescriptorSetUpdater &write_acceleration_structure(
        VkDescriptorSet set, uint32_t binding, const std::unique_ptr<TopLevelBVH> &bvh);

    DescriptorSetUpdater &write_storage_image(VkDescriptorSet set,
                                              uint32_t binding,
                                              const std::shared_ptr<Texture2D> &img);

    DescriptorSetUpdater &write_ubo(VkDescriptorSet set,
                                    uint32_t binding,
                                    const std::shared_ptr<Buffer> &buf);

    DescriptorSetUpdater &write_ssbo(VkDescriptorSet set,
                                     uint32_t binding,
                                     const std::shared_ptr<Buffer> &buf);

    DescriptorSetUpdater &write_ssbo_array(VkDescriptorSet set,
                                           uint32_t binding,
                                           const std::vector<std::shared_ptr<Buffer>> &bufs);

    DescriptorSetUpdater &write_combined_sampler_array(
        VkDescriptorSet set,
        uint32_t binding,
        const std::vector<CombinedImageSampler> &combined_samplers);

    // Commit the writes to the descriptor sets
    void update(Device &device);
};

}
