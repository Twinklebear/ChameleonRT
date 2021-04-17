#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

// Quad-shaped light source
struct QuadLight {
    vec4 emission;
    vec4 position;
    vec4 normal;
    // x and y vectors spanning the quad, with
    // the half-width and height in the w component
    vec4 v_x;
    vec4 v_y;
};

vec3 sample_quad_light_position(in const QuadLight light, vec2 samples)
{
    return samples.x * light.v_x.xyz * light.v_x.w + samples.y * light.v_y.xyz * light.v_y.w +
           light.position.xyz;
}

/* Compute the PDF of sampling the sampled point p light with the ray specified by orig and
 * dir, assuming the light is not occluded
 */
float quad_light_pdf(in const QuadLight light,
                     in const vec3 p,
                     in const vec3 orig,
                     in const vec3 dir)
{
    float surface_area = light.v_x.w * light.v_y.w;
    vec3 to_pt = p - dir;
    float dist_sqr = dot(to_pt, to_pt);
    float n_dot_w = dot(light.normal.xyz, -dir);
    if (n_dot_w < EPSILON) {
        return 0.f;
    }
    return dist_sqr / (n_dot_w * surface_area);
}

bool quad_intersect(in const QuadLight light,
                    in const vec3 orig,
                    in const vec3 dir,
                    out float t,
                    out vec3 light_pos)
{
    float denom = dot(dir, light.normal.xyz);
    if (denom != 0.f) {
        t = dot(light.position.xyz - orig, light.normal.xyz) / denom;
        if (t < 0.f) {
            return false;
        }

        // It's a finite plane so now see if the hit point is actually inside the plane
        light_pos = orig + dir * t;
        vec3 hit_v = light_pos - light.position.xyz;
        if (abs(dot(hit_v, light.v_x.xyz)) < light.v_x.w &&
            abs(dot(hit_v, light.v_y.xyz)) < light.v_y.w) {
            return true;
        }
    }
    return false;
}

#endif

