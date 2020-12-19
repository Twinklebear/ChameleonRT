#include "metaldisplay.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <Cocoa/Cocoa.h>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.h>
#include "display/imgui_impl_sdl.h"
#include "imgui_impl_metal.h"
#include "metaldisplay_embedded_metallib.h"
#include "metalrt_utils.h"
#include "util.h"

struct MetalDisplayData {
    CAMetalLayer *layer = nullptr;
    MTLRenderPassDescriptor *render_pass = nullptr;
    id<CAMetalDrawable> current_drawable = nullptr;
};

MetalDisplay::MetalDisplay(SDL_Window *window)
    : context(std::make_shared<metal::Context>()), data(std::make_shared<MetalDisplayData>())
{
    @autoreleasepool {
        SDL_SysWMinfo wm_info;
        SDL_VERSION(&wm_info.version);
        SDL_GetWindowWMInfo(window, &wm_info);

        // Setup the Metal layer
        data->layer = [CAMetalLayer layer];
        data->layer.device = context->device;
        data->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        data->layer.framebufferOnly = NO;

        NSWindow *nswindow = wm_info.info.cocoa.window;
        nswindow.contentView.layer = data->layer;
        nswindow.contentView.wantsLayer = YES;

        // We need to use a compute pipeline to do the RGBA->BGRA swizzle, since Metla doesn't
        // allow RGBA8 display pixel formats
        shader_library = std::make_shared<metal::ShaderLibrary>(
            *context, metaldisplay_metallib, sizeof(metaldisplay_metallib));

        pipeline = std::make_shared<metal::ComputePipeline>(
            *context, shader_library->new_function(@"display_image"));

        data->render_pass = [MTLRenderPassDescriptor new];
        data->render_pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        data->render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        data->render_pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

        ImGui_ImplMetal_Init(context->device);
        ImGui_ImplSDL2_InitForMetal(window);
    }
}

MetalDisplay::~MetalDisplay()
{
    ImGui_ImplMetal_Shutdown();
}

std::string MetalDisplay::gpu_brand()
{
    return context->device_name();
}

std::string MetalDisplay::name()
{
    return "Metal";
}

void MetalDisplay::resize(const int fb_width, const int fb_height)
{
    @autoreleasepool {
        fb_dims = glm::uvec2(fb_width, fb_height);
        upload_texture = std::make_shared<metal::Texture2D>(*context,
                                                            fb_width,
                                                            fb_height,
                                                            MTLPixelFormatRGBA8Unorm,
                                                            MTLTextureUsageShaderRead);
    }
}

void MetalDisplay::new_frame()
{
    @autoreleasepool {
        data->current_drawable = [data->layer nextDrawable];
        data->render_pass.colorAttachments[0].texture = data->current_drawable.texture;
        ImGui_ImplMetal_NewFrame(data->render_pass);
    }
}

void MetalDisplay::display(const std::vector<uint32_t> &img)
{
    upload_texture->upload(img.data());
    display_native(upload_texture);
}

void MetalDisplay::display_native(std::shared_ptr<metal::Texture2D> &img)
{
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = context->command_buffer();

        // Copy the rendered image to the framebuffer
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
        [command_encoder setTexture:img->texture atIndex:0];
        [command_encoder setTexture:data->current_drawable.texture atIndex:1];

        // Display the image
        [command_encoder setComputePipelineState:pipeline->pipeline];
        [command_encoder dispatchThreads:MTLSizeMake(fb_dims.x, fb_dims.y, 1)
                   threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

        [command_encoder endEncoding];

        // Draw ImGui on top
        // TODO: Hitting some weird crash in ImGui about rasterSampleCount (0) not supported?
        // Doesn't happen in the examples and I can't see where my config is going wrong
        id<MTLRenderCommandEncoder> imgui_encoder =
            [command_buffer renderCommandEncoderWithDescriptor:data->render_pass];
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), command_buffer, imgui_encoder);
        [imgui_encoder endEncoding];

        [command_buffer presentDrawable:data->current_drawable];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        data->current_drawable = nullptr;
    }
}

