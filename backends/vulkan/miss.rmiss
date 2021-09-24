#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT RayPayload payload;

void main() {
    payload.dist = -1;

    vec3 dir = gl_WorldRayDirectionEXT;
    float u = (1.f + atan(dir.x, -dir.z) * M_1_PI) * 0.5f;
    float v = acos(dir.y) * M_1_PI;

    int check_x = int(u * 10.f);
    int check_y = int(v * 10.f);

    if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
        payload.normal.rgb = vec3(0.5f);
    } else {
        payload.normal.rgb = vec3(0.1f);
    }
}

