#ifndef UTIL_GLSL
#define UTIL_GLSL

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference2 : enable

#define M_PI 3.14159265358979323846f
#define M_1_PI 0.318309886183790671538f
#define EPSILON 0.0001f
#define uint32_t uint

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define MAX_PATH_DEPTH 5

struct RayPayload {
    vec3 normal;
    float dist;
    vec2 uv;
    uint material_id;
    float pad;
};

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

void ortho_basis(out vec3 v_x, out vec3 v_y, const vec3 n) {
	v_y = vec3(0, 0, 0);

	if (n.x < 0.6f && n.x > -0.6f) {
		v_y.x = 1.f;
	} else if (n.y < 0.6f && n.y > -0.6f) {
		v_y.y = 1.f;
	} else if (n.z < 0.6f && n.z > -0.6f) {
		v_y.z = 1.f;
	} else {
		v_y.x = 1.f;
	}
	v_x = normalize(cross(v_y, n));
	v_y = normalize(cross(n, v_x));
}

float luminance(in const vec3 c) {
	return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

float pow2(float x) {
	return x * x;
}

#endif

