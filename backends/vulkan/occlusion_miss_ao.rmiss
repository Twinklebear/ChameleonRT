#version 460

#include "util.glsl"

layout(location = OCCLUSION_RAY) rayPayloadInEXT int n_occluded;

void main() {
    --n_occluded;
}

