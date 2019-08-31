#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInNV RayPayload payload;

void main() {
    payload.dist = -1;
}

