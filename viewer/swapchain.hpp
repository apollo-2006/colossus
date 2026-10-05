#pragma once
// the window's swapchain, remade on resize.
#include "vk.hpp"

#include <vector>

namespace viewer {

class swapchain {
public:
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> done;  // one per image: presenting waits on it
    VkExtent2D extent{};

    swapchain(vk::context& ctx, bool vsync) : ctx_(ctx), vsync_(vsync) {}
    ~swapchain() { destroy(); }

    void create(uint32_t width, uint32_t height) {
        VkSurfaceCapabilitiesKHR caps;
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.gpu, ctx_.surface, &caps));
        extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{width, height};
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.gpu, ctx_.surface, &n, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.gpu, ctx_.surface, &n, formats.data());
        VkSurfaceFormatKHR format = formats[0];
        for (const auto& f : formats)
            if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                format = f;
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.gpu, ctx_.surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.gpu, ctx_.surface, &n, modes.data());
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if (!vsync_)
            for (VkPresentModeKHR m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR || (m == VK_PRESENT_MODE_MAILBOX_KHR && mode != VK_PRESENT_MODE_IMMEDIATE_KHR))
                    mode = m;

        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = ctx_.surface;
        sci.minImageCount = std::max(caps.minImageCount, 3u);
        if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
        sci.imageFormat = format.format;
        sci.imageColorSpace = format.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = mode;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = handle;
        VkSwapchainKHR fresh;
        VK_CHECK(vkCreateSwapchainKHR(ctx_.device, &sci, nullptr, &fresh));
        destroy();
        handle = fresh;
        vkGetSwapchainImagesKHR(ctx_.device, handle, &n, nullptr);
        images.resize(n);
        vkGetSwapchainImagesKHR(ctx_.device, handle, &n, images.data());
        done.resize(n);
        for (VkSemaphore& s : done) {
            VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(ctx_.device, &ci, nullptr, &s));
        }
    }

private:
    vk::context& ctx_;
    bool vsync_;

    void destroy() {
        if (!handle) return;
        vkDeviceWaitIdle(ctx_.device);
        for (VkSemaphore s : done) vkDestroySemaphore(ctx_.device, s, nullptr);
        done.clear();
        vkDestroySwapchainKHR(ctx_.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
};

}  // namespace viewer
