#version 460
#extension GL_NV_ray_tracing : require

#include "types.glsl"

layout(location = 0) rayPayloadInNV RayPayload payload;

void main() {
    payload.dist = -1;
}

