#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT AORayPayload payload;

void main() {
    payload.dist = -1;
}

