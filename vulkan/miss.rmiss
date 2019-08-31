#version 460
#extension GL_NV_ray_tracing : require

layout(location = 0) rayPayloadInNV vec4 hit_color;

void main() {
	hit_color = vec4(0, 0, 0, -1);
}

