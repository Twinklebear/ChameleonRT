#version 460
#extension GL_NV_ray_tracing : require

layout(location = 0) rayPayloadInNV vec4 hitColor;

void main() {
	hitColor = vec4(0, 0, 1, -1);
}

