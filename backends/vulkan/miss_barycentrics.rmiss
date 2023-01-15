#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT BarycentricsPayload payload;

void main() {
    payload.dist = -1;
}

