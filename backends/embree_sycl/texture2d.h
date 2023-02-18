#pragma once

#include "float3.h"
#include "util.h"

namespace kernel {

inline float4 get_texel(const embree::ISPCTexture2D *tex, const int2 px)
{
    float4 color = make_float4(0.f);
    color.x = tex->data[((px.y * tex->width) + px.x) * tex->channels] / 255.f;
    if (tex->channels >= 2) {
        color.y = tex->data[((px.y * tex->width) + px.x) * tex->channels + 1] / 255.f;
    }
    if (tex->channels >= 3) {
        color.z = tex->data[((px.y * tex->width) + px.x) * tex->channels + 2] / 255.f;
    }
    if (tex->channels == 4) {
        color.w = tex->data[((px.y * tex->width) + px.x) * tex->channels + 3] / 255.f;
    }
    return color;
}

inline float get_texel_channel(const embree::ISPCTexture2D *tex,
                               const int2 px,
                               const int channel)
{
    return tex->data[((px.y * tex->width) + px.x) * tex->channels + channel] / 255.f;
}

int mod(int a, int b)
{
    if (b == 0) {
        b = 1;
    }
    int r = a - (a / b) * b;
    return r < 0 ? r + b : r;
}

inline int2 get_wrapped_texcoord(const embree::ISPCTexture2D *tex, int x, int y)
{
    int w = tex->width;
    int h = tex->height;
    // TODO: maybe support other wrap modes?
    return make_int2(mod(x, w), mod(y, h));
}

float4 texture(const embree::ISPCTexture2D *tex, const float2 uv)
{
    const float ux = uv.x * tex->width - 0.5f;
    const float uy = uv.y * tex->height - 0.5f;

    const float tx = ux - sycl::floor(ux);
    const float ty = uy - sycl::floor(uy);

    const int2 t00 = get_wrapped_texcoord(tex, ux, uy);
    const int2 t10 = get_wrapped_texcoord(tex, ux + 1.f, uy);
    const int2 t01 = get_wrapped_texcoord(tex, ux, uy + 1.f);
    const int2 t11 = get_wrapped_texcoord(tex, ux + 1.f, uy + 1.f);

    const float4 s00 = get_texel(tex, t00);
    const float4 s10 = get_texel(tex, t10);
    const float4 s01 = get_texel(tex, t01);
    const float4 s11 = get_texel(tex, t11);

    return s00 * (1.f - tx) * (1.f - ty) + s10 * tx * (1.f - ty) + s01 * (1.f - tx) * ty +
           s11 * tx * ty;
}

float texture_channel(const embree::ISPCTexture2D *tex, const float2 uv, const int channel)
{
    const float ux = uv.x * tex->width - 0.5f;
    const float uy = uv.y * tex->height - 0.5f;

    const float tx = ux - sycl::floor(ux);
    const float ty = uy - sycl::floor(uy);

    const int2 t00 = get_wrapped_texcoord(tex, ux, uy);
    const int2 t10 = get_wrapped_texcoord(tex, ux + 1.f, uy);
    const int2 t01 = get_wrapped_texcoord(tex, ux, uy + 1.f);
    const int2 t11 = get_wrapped_texcoord(tex, ux + 1.f, uy + 1.f);

    const float s00 = get_texel_channel(tex, t00, channel);
    const float s10 = get_texel_channel(tex, t10, channel);
    const float s01 = get_texel_channel(tex, t01, channel);
    const float s11 = get_texel_channel(tex, t11, channel);

    return s00 * (1.f - tx) * (1.f - ty) + s10 * tx * (1.f - ty) + s01 * (1.f - tx) * ty +
           s11 * tx * ty;
}

}
