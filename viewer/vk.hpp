#pragma once
// A thin layer over Vulkan for the viewer: one device and queue, buffers,
// images, shader modules, one-shot submissions, and a swapchain when there
// is a window. Grown from photon_tracer's gpu/vk.hpp.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(call)                                                                                     \
    do {                                                                                                   \
        const VkResult result_ = (call);                                                                   \
        if (result_ != VK_SUCCESS)                                                                         \
            throw std::runtime_error(std::string(#call) + " failed: VkResult " + std::to_string(result_)); \
    } while (0)

namespace vk {

inline int validation_errors = 0;

inline VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT,
                                                 const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "validation: %s\n", data->pMessage);
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++validation_errors;
    }
    return VK_FALSE;
}

struct buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // For host-visible buffers
};

struct image {
    VkImage handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};

class context {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::string device_name;

    PFN_vkCmdDrawMeshTasksIndirectEXT draw_mesh_tasks_indirect = nullptr;

    // Ray queries, if the GPU has them: the viewer then traces shadows.
    bool ray_query = false;
    PFN_vkCreateAccelerationStructureKHR create_as = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy_as = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR as_build_sizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR build_as = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR as_address = nullptr;

    // With a window, the instance gets the extensions GLFW needs and the
    // device a swapchain.
    context(GLFWwindow* window, bool validate) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "nexus_geometry";
        app.apiVersion = VK_API_VERSION_1_3;
        std::vector<const char*> layers, extensions;
        if (window) {
            uint32_t n = 0;
            const char** glfw_ext = glfwGetRequiredInstanceExtensions(&n);
            extensions.assign(glfw_ext, glfw_ext + n);
        }
        if (validate) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ici.ppEnabledLayerNames = layers.data();
        ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ici.ppEnabledExtensionNames = extensions.data();
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
        if (validate) {
            VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
            mci.pfnUserCallback = on_message;
            auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            if (create) VK_CHECK(create(instance, &mci, nullptr, &messenger));
        }
        if (window) VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
        pick_device();
        create_device();
    }

    ~context() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (messenger) {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy) destroy(instance, messenger, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    context(const context&) = delete;
    context& operator=(const context&) = delete;

    buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
        buffer b;
        b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size ? size : 16;
        bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &b.handle));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.handle, &req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flags;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, host_visible
            ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &b.memory));
        VK_CHECK(vkBindBufferMemory(device, b.handle, b.memory, 0));
        if (host_visible) VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = b.handle;
        b.address = vkGetBufferDeviceAddress(device, &bai);
        return b;
    }

    // A device-local buffer holding bytes, copied in through a staging
    // buffer in pieces, so a model of any size needs little host memory.
    buffer upload(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
        buffer b = make_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        const VkDeviceSize piece = 64ull << 20;
        buffer staging = make_buffer(std::min(size, piece), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        for (VkDeviceSize at = 0; at < size; at += piece) {
            const VkDeviceSize n = std::min(piece, size - at);
            std::memcpy(staging.mapped, static_cast<const char*>(data) + at, n);
            submit([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{0, at, n};
                vkCmdCopyBuffer(cmd, staging.handle, b.handle, 1, &copy);
            });
        }
        destroy(staging);
        return b;
    }
    template <class T>
    buffer upload(const std::vector<T>& v, VkBufferUsageFlags usage) {
        return upload(v.data(), v.size() * sizeof(T), usage);
    }

    image make_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        image im;
        im.format = format;
        im.width = w;
        im.height = h;
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = format;
        ic.extent = {w, h, 1};
        ic.mipLevels = 1;
        ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = usage;
        VK_CHECK(vkCreateImage(device, &ic, nullptr, &im.handle));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, im.handle, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &im.memory));
        VK_CHECK(vkBindImageMemory(device, im.handle, im.memory, 0));
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = im.handle;
        vc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vc.format = format;
        vc.subresourceRange = {aspect, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device, &vc, nullptr, &im.view));
        return im;
    }

    // An image from a full description, with a view of all its levels.
    image make_image(const VkImageCreateInfo& ic, VkImageAspectFlags aspect) {
        image im;
        im.format = ic.format;
        im.width = ic.extent.width;
        im.height = ic.extent.height;
        VK_CHECK(vkCreateImage(device, &ic, nullptr, &im.handle));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, im.handle, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &im.memory));
        VK_CHECK(vkBindImageMemory(device, im.handle, im.memory, 0));
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = im.handle;
        vc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vc.format = ic.format;
        vc.subresourceRange = {aspect, 0, ic.mipLevels, 0, 1};
        VK_CHECK(vkCreateImageView(device, &vc, nullptr, &im.view));
        return im;
    }

    void destroy(buffer& b) {
        if (!b.handle) return;
        if (b.mapped) vkUnmapMemory(device, b.memory);
        vkDestroyBuffer(device, b.handle, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
        b = buffer{};
    }
    void destroy(image& im) {
        if (!im.handle) return;
        vkDestroyImageView(device, im.view, nullptr);
        vkDestroyImage(device, im.handle, nullptr);
        vkFreeMemory(device, im.memory, nullptr);
        im = image{};
    }

    VkCommandBuffer allocate_command_buffer() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
        return cmd;
    }

    // Records commands with record(cmd), submits them and waits.
    template <class Record>
    void submit(Record&& record) {
        VkCommandBuffer cmd = allocate_command_buffer();
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        record(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(device, pool, 1, &cmd);
    }

    VkShaderModule shader(const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = bytes;
        sci.pCode = code;
        VkShaderModule m;
        VK_CHECK(vkCreateShaderModule(device, &sci, nullptr, &m));
        return m;
    }

private:
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

    std::vector<const char*> required() const {
        std::vector<const char*> r = {VK_EXT_MESH_SHADER_EXTENSION_NAME};
        if (surface) r.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        return r;
    }

    // The first GPU with Vulkan 1.3 and mesh shaders (and presenting, with
    // a window); discrete first.
    void pick_device() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> gpus(n);
        vkEnumeratePhysicalDevices(instance, &n, gpus.data());
        VkPhysicalDevice fallback = VK_NULL_HANDLE;
        for (VkPhysicalDevice d : gpus) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.apiVersion < VK_API_VERSION_1_3 || !has_extensions(d)) continue;
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { gpu = d; break; }
            if (!fallback) fallback = d;
        }
        if (!gpu) gpu = fallback;
        if (!gpu) throw std::runtime_error("no GPU with Vulkan 1.3 and VK_EXT_mesh_shader");
        vkGetPhysicalDeviceProperties(gpu, &properties);
        device_name = properties.deviceName;
    }

    static constexpr const char* ray_query_extensions[] = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };

    static bool has(VkPhysicalDevice d, const char* const* names, size_t count) {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> ext(n);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &n, ext.data());
        for (size_t i = 0; i < count; ++i) {
            bool found = false;
            for (const auto& e : ext) found |= std::strcmp(e.extensionName, names[i]) == 0;
            if (!found) return false;
        }
        return true;
    }

    bool has_extensions(VkPhysicalDevice d) const {
        const auto r = required();
        return has(d, r.data(), r.size());
    }

    void create_device() {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, nullptr);
        std::vector<VkQueueFamilyProperties> families(n);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, families.data());
        queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < n; ++i) {
            const VkQueueFlags want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if ((families[i].queueFlags & want) != want) continue;
            VkBool32 present = VK_TRUE;
            if (surface) vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &present);
            if (present) { queue_family = i; break; }
        }
        if (queue_family == UINT32_MAX) throw std::runtime_error("no graphics, compute and present queue");

        const float priority = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        ray_query = has(gpu, ray_query_extensions, std::size(ray_query_extensions)) && !std::getenv("NEXUS_NO_RAY_QUERY");
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        rq.rayQuery = VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        as.accelerationStructure = VK_TRUE;
        as.pNext = &rq;
        VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        mesh.taskShader = VK_TRUE;
        mesh.meshShader = VK_TRUE;
        if (ray_query) mesh.pNext = &as;
        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13.dynamicRendering = VK_TRUE;
        v13.synchronization2 = VK_TRUE;
        v13.maintenance4 = VK_TRUE;
        v13.pNext = &mesh;
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        v12.scalarBlockLayout = VK_TRUE;
        v12.shaderBufferInt64Atomics = VK_TRUE;  // The visibility buffer
        v12.hostQueryReset = VK_TRUE;
        v12.bufferDeviceAddress = VK_TRUE;
        v12.pNext = &v13;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.features.shaderInt64 = VK_TRUE;
        features.features.fragmentStoresAndAtomics = VK_TRUE;
        features.features.multiDrawIndirect = VK_TRUE;
        features.features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;  // The depth pyramid's levels
        features.pNext = &v12;

        auto ext = required();
        if (ray_query) ext.insert(ext.end(), std::begin(ray_query_extensions), std::end(ray_query_extensions));
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &features;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(ext.size());
        dci.ppEnabledExtensionNames = ext.data();
        VK_CHECK(vkCreateDevice(gpu, &dci, nullptr, &device));
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = queue_family;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

        draw_mesh_tasks_indirect = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectEXT"));
        if (ray_query) {
            create_as = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
            destroy_as = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
            as_build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
            build_as = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
            as_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
        }
    }

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("no suitable memory type");
    }
};

// Records a memory barrier between two stages.
inline void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = src_stage;
    mb.srcAccessMask = src_access;
    mb.dstStageMask = dst_stage;
    mb.dstAccessMask = dst_access;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// Moves an image between layouts, waiting on src_stage and blocking dst_stage.
inline void transition(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                       VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask = src_stage;
    ib.srcAccessMask = src_access;
    ib.dstStageMask = dst_stage;
    ib.dstAccessMask = dst_access;
    ib.oldLayout = from;
    ib.newLayout = to;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = image;
    ib.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(cmd, &di);
}

}  // namespace vk
