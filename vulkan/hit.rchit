#version 460
#extension GL_NV_ray_tracing : require

layout(location = 0) rayPayloadInNV vec4 hitColor;

layout(shaderRecordNV) buffer SBT {
	float blueness;
};

void main() {
	hitColor = vec4(0, 0, blueness, 1.f);
}

