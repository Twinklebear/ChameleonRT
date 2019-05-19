#define M_PI 3.14159265358979323846
#define M_1_PI 0.318309886183790671538

struct HitInfo {
	float4 color_dist;
	float4 normal;
};

struct OcclusionHitInfo {
	int hit;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
	float2 bary;
};

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> output : register(u0);

// Accumulation buffer for progressive refinement
RWTexture2D<float4> accum_buffer : register(u1);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

// View params buffer
cbuffer ViewParams : register(b0) {
	float4 cam_pos;
	float4 cam_du;
	float4 cam_dv;
	float4 cam_dir_top_left;
	uint32_t frame_id;
}

uint32_t pcg32_random(inout PCGRand rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005ULL + 1;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float pcg32_randomf(inout PCGRand rng) {
	return ldexp((double)pcg32_random(rng), -32);
}

PCGRand get_rng() {
	uint2 pixel = DispatchRaysIndex().xy;
	uint32_t seed = (pixel.x + pixel.y * DispatchRaysDimensions().x) * (frame_id + 1);

	PCGRand rng;
	rng.state = 0;
	pcg32_random(rng);
	rng.state += seed;
	pcg32_random(rng);
	return rng;
}

float linear_to_srgb(float x) {
	if (x <= 0.0031308f) {
		return 12.92f * x;
	}
	return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

void ortho_basis(out float3 v_x, out float3 v_y, const float3 n) {
	v_y = float3(0, 0, 0);

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

/* Disney BRDF functions, for additional details and examples see:
 * - https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
 * - https://www.shadertoy.com/view/XdyyDd
 * - https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
 * - https://schuttejoe.github.io/post/disneybsdf/
 *
 * Variable naming conventions with the Burley course notes:
 * V -> w_o
 * L -> w_i
 * H -> w_h
 */

struct DisneyMaterial {
	float3 base_color;
	float metallic;
	float specular;
	float roughness;
	float specular_tint;
	float anisotropy;
	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;
};

cbuffer MaterialParams : register(b1) {
	float4 basecolor_metallic;
	float4 spec_rough_spectint_aniso;
	float4 sheen_sheentint_clearc_ccgloss;
}

bool same_hemisphere(in const float3 w_o, in const float3 w_i, in const float3 n) {
	return dot(w_o, n) * dot(w_i, n) > 0.f;
}

// Sample the hemisphere using a cosine weighted distribution,
// returns a vector in a hemisphere oriented about (0, 0, 1)
float3 cos_sample_hemisphere(float2 u) {
	float2 s = 2.f * u - 1.f;
	float2 d;
	float radius = 0;
	float theta = 0;
    if (s.x == 0.f && s.y == 0.f) {
        d = s;
    } else {
		if (abs(s.x) > abs(s.y)) {
			radius = s.x;
			theta  = M_PI / 4.f * (s.y / s.x);
		} else {
			radius = s.y;
			theta  = M_PI / 2.f - M_PI / 4.f * (s.x / s.y);
		}
	}
	d = radius * float2(cos(theta), sin(theta));
	return float3(d.x, d.y, sqrt(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
}

float luminance(in const float3 c) {
	// I wonder why they use this approximate luminance rather than the equation on wikipedia?
	return 0.3f * c.r + 0.6f * c.g + 0.1f * c.b;
	// Wikipedia luminance:
	//return 0.2126f * mat.base_color.r + 0.7152f * mat.base_color.g + 0.0722f * mat.base_color.b;
}

float schlick_weight(float cos_theta) {
	return pow(saturate(1.f - cos_theta), 5.f);
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
// Burley notes eq. 4
float gtr_1(float cos_theta_h, float alpha) {
	if (alpha >= 1.f) {
		return M_1_PI;
	}
	float alpha_sqr = alpha * alpha;
	return M_1_PI * (alpha_sqr - 1.f) / (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 8
float gtr_2(float cos_theta_h, float alpha) {
	float alpha_sqr = alpha * alpha;
	return M_1_PI * alpha_sqr / pow(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h, 2.f);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, float2 alpha) {
	return M_1_PI / (alpha.x * alpha.y
			* pow(pow(h_dot_x / alpha.x, 2.f) + pow(h_dot_y / alpha.y, 2.f) + h_dot_n * h_dot_n, 2.f));
}

float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
	float a = alpha_g * alpha_g;
	float b = n_dot_o * n_dot_o;
	return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, float2 alpha) {
	return 1.f / (n_dot_o + sqrt(pow(o_dot_x * alpha.x, 2.f) + pow(o_dot_y * alpha.y, 2.f) + pow(n_dot_o, 2.f)));
}

// Sample the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
float3 sample_lambertian(in const float3 n, in const float3 v_x, in const float3 v_y, in const float2 s) {
	const float3 hemi_dir = normalize(cos_sample_hemisphere(s));
	return normalize(hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n);
}

float lambertian_pdf(in const float3 w_i, in const float3 n) {
	float d = dot(w_i, n);
	if (d > 0.f) {
		return d * M_1_PI;
	}
	return 0.f;
}

float gtr_1_pdf(in const float3 w_o, in const float3 w_h, in const float3 w_i, in const float3 n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float cos_theta_h = dot(n, w_h);
	float d = gtr_1(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float gtr_2_pdf(in const float3 w_o, in const float3 w_h, in const float3 w_i, in const float3 n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float gtr_2_aniso_pdf(in const float3 w_o, in const float3 w_h, in const float3 w_i, in const float3 n,
	in const float3 v_x, in const float3 v_y, const float2 alpha) 
{
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2_aniso(cos_theta_h, abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float3 disney_diffuse(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h)
{
	float n_dot_o = abs(dot(w_o, n));
	float n_dot_i = abs(dot(w_i, n));
	float fd90 = 0.5f + 2.f * mat.roughness * pow(dot(w_i, w_h), 2.0);
	float fi = schlick_weight(n_dot_i);
	float fo = schlick_weight(n_dot_o);
	return mat.base_color * M_1_PI * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

float3 disney_microfacet_isotropic(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h)
{
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : float3(1, 1, 1);
	float3 spec = lerp(mat.specular * 0.08 * lerp(float3(1, 1, 1), tint, mat.specular_tint), mat.base_color, mat.metallic);

	float alpha = max(0.001, mat.roughness * mat.roughness);
	float d = gtr_2(dot(n, w_h), alpha);
	float3 f = lerp(spec, float3(1, 1, 1), schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx(dot(n, w_i), alpha) * smith_shadowing_ggx(dot(n, w_o), alpha);
	return d * f * g;
}

float3 disney_microfacet_anisotropic(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h, in const float3 v_x, in const float3 v_y)
{
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : float3(1, 1, 1);
	float3 spec = lerp(mat.specular * 0.08 * lerp(float3(1, 1, 1), tint, mat.specular_tint), mat.base_color, mat.metallic);

	float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
	float a = mat.roughness * mat.roughness;
	float2 alpha = float2(max(0.001, a / aspect), max(0.001, a * aspect));
	float d = gtr_2_aniso(dot(n, w_h), abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
	float3 f = lerp(spec, float3(1, 1, 1), schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx_aniso(dot(n, w_i), abs(dot(w_i, v_x)), abs(dot(w_i, v_y)), alpha)
		* smith_shadowing_ggx_aniso(dot(n, w_o), abs(dot(w_o, v_x)), abs(dot(w_o, v_y)), alpha);
	return d * f * g;
}

float disney_clear_coat(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h)
{
	float d = gtr_1(dot(n, w_h), lerp(0.1f, 0.001f, mat.clearcoat_gloss));
	float f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx(dot(n, w_i), 0.25f) * smith_shadowing_ggx(dot(n, w_o), 0.25f);
	return 0.25 * mat.clearcoat * d * f * g;
}

float3 disney_sheen(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h)
{
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : float3(1, 1, 1);
	float3 sheen_color = lerp(float3(1, 1, 1), tint, mat.sheen_tint);
	float f = schlick_weight(dot(w_i, w_h));
	return f * mat.sheen * sheen_color;
}

float3 disney_brdf(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h,
	in const float3 v_x, in const float3 v_y)
{
	if (!same_hemisphere(w_o, w_i, n)) {
		return float3(0, 0, 0);
	}

	float3 diffuse = disney_diffuse(mat, n, w_o, w_i, w_h);
	float3 gloss;
	if (mat.anisotropy == 0.f) {
		gloss = disney_microfacet_isotropic(mat, n, w_o, w_i, w_h);
	} else {
		gloss = disney_microfacet_anisotropic(mat, n, w_o, w_i, w_h, v_x, v_y);
	}
	float coat = disney_clear_coat(mat, n, w_o, w_i, w_h);
	float3 sheen = disney_sheen(mat, n, w_o, w_i, w_h);
	return (diffuse + sheen) * (1.f - mat.metallic) + gloss + coat;
}

float disney_pdf(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 w_h,
	in const float3 v_x, in const float3 v_y)
{
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0;
	}

	float alpha = max(0.001, mat.roughness * mat.roughness);
	float clearcoat_alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);

	float diffuse = lambertian_pdf(w_i, n);
	float microfacet;
	if (mat.anisotropy == 0.f) {
		microfacet = gtr_2_pdf(w_o, w_h, w_i, n, alpha);
	} else {
		microfacet = gtr_2_aniso_pdf(w_o, w_h, w_i, n, v_x, v_y, alpha);
	}
	float clear_coat = gtr_1_pdf(w_o, w_h, w_i, n, clearcoat_alpha);
	// TODO: We only need to do the division by 3 when we are actually sampling
	// a specific material layer to go through. Right now it's essentially always
	// picking the diffuse layer to sample, so we just weight by that
	//return (diffuse + microfacet + clear_coat);// / 3.f;
	return diffuse;
}

[shader("raygeneration")] 
void RayGen() {
	const int PRIMARY_RAY = 0;
	const int OCCLUSION_RAY = 1;
	const int NUM_RAY_TYPES = 2;
	const int MAX_PATH_DEPTH = 5;

	uint2 pixel = DispatchRaysIndex().xy;
	float2 dims = float2(DispatchRaysDimensions().xy);
	PCGRand rng = get_rng();
	float2 d = (pixel + float2(pcg32_randomf(rng), pcg32_randomf(rng))) / dims;

	RayDesc ray;
	ray.Origin = cam_pos.xyz;
	ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
	ray.TMin = 0;
	ray.TMax = 1e20f;

	DisneyMaterial mat;
	mat.base_color = basecolor_metallic.rgb;
	mat.metallic = basecolor_metallic.a;
	mat.specular = spec_rough_spectint_aniso.r;
	mat.roughness = spec_rough_spectint_aniso.g;
	mat.specular_tint = spec_rough_spectint_aniso.b;
	mat.anisotropy = spec_rough_spectint_aniso.a;
	mat.sheen = sheen_sheentint_clearc_ccgloss.r;
	mat.sheen_tint = sheen_sheentint_clearc_ccgloss.g;
	mat.clearcoat = sheen_sheentint_clearc_ccgloss.b;
	mat.clearcoat_gloss = sheen_sheentint_clearc_ccgloss.a;

	const float3 light_emission = float3(1.0, 1.0, 1.0);
	int bounce = 0;
	float3 illum = float3(0, 0, 0);
	float3 path_throughput = float3(1, 1, 1);
	do {
		HitInfo payload;
		payload.color_dist = float4(0, 0, 0, -1);
		TraceRay(scene, 0, 0xff, PRIMARY_RAY, NUM_RAY_TYPES, PRIMARY_RAY, ray, payload);

		if (payload.color_dist.w <= 0) {
			break;
		}

		const float3 w_o = -ray.Direction;
		const float3 hit_p = ray.Origin + payload.color_dist.w * ray.Direction;
		float3 v_x, v_y;
		float3 v_z = normalize(payload.normal.xyz);
		if (dot(v_z, w_o) < 0.0) {
			v_z = -v_z;
		}
		ortho_basis(v_x, v_y, v_z);

		const float roughness = 0.f;

		// Direct light sampling.
		// TODO: should sample the microfacet distribution
		const float3 light_dir = normalize(float3(-0.5, 0.8, 0.5));
		float3 w_h = normalize(w_o + light_dir);
		float3 bsdf = disney_brdf(mat, v_z, w_o, light_dir, w_h, v_x, v_y);

		OcclusionHitInfo shadow_hit;
		RayDesc shadow_ray;
		shadow_ray.Origin = hit_p;
		shadow_ray.Direction = light_dir;
		shadow_ray.TMin = 0.0001;
		shadow_ray.TMax = 1e20f;

		TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
				OCCLUSION_RAY, NUM_RAY_TYPES, OCCLUSION_RAY, shadow_ray, shadow_hit);
		if (shadow_hit.hit == 0) {
			// Light is a delta light, so pdf = 1.0
			illum += path_throughput * bsdf * light_emission * abs(dot(light_dir, v_z));
		}

		// Sample the hemisphere to compute the outgoing direction
		// TODO: We need to use the BRDF to do the sampling, otherwise for super smooth
		// materials we do a terrible job
		float3 w_i = sample_lambertian(v_z, v_x, v_y, float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		w_h = normalize(w_o + w_i);

		// Update path throughput and continue the ray
		float pdf = disney_pdf(mat, v_z, w_o, w_i, w_h, v_x, v_y);
		if (pdf < 0.0001) {
			break;
		}

		bsdf = disney_brdf(mat, v_z, w_o, w_i, w_h, v_x, v_y);
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

		if (path_throughput.x < 0.0001 && path_throughput.y < 0.0001 && path_throughput.z < 0.0001) {
			break;
		}

		ray.Origin = hit_p;
		ray.Direction = w_i;
		ray.TMin = 0.0001;
		ray.TMax = 1e20f;

		++bounce;
	} while (bounce < MAX_PATH_DEPTH);

	const float4 accum_color = (float4(illum, 1.0) + frame_id * accum_buffer[pixel]) / (frame_id + 1);
	accum_buffer[pixel] = accum_color;

	output[pixel] = float4(linear_to_srgb(accum_color.r),
		linear_to_srgb(accum_color.g),
		linear_to_srgb(accum_color.b), 1.f);
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload) {
	payload.color_dist = float4(0, 0, 0, 0);
}

[shader("miss")]
void AOMiss(inout OcclusionHitInfo occlusion : SV_RayPayload) {
	occlusion.hit = 0;
}

StructuredBuffer<float3> vertices : register(t0, space1);
StructuredBuffer<uint3> indices : register(t1, space1);

[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) {
	uint3 idx = indices[PrimitiveIndex()];
	float3 va = vertices[idx.x];
	float3 vb = vertices[idx.y];
	float3 vc = vertices[idx.z];
	float3 n = normalize(cross(vb - va, vc - va));
	payload.color_dist = float4(0.9, 0.9, 0.9, RayTCurrent());
	payload.normal = float4(n, 0);
}

[shader("closesthit")]
void AOHit(inout OcclusionHitInfo occlusion, Attributes attrib) {
	occlusion.hit = 1;
}
