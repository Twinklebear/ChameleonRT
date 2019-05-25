#define M_PI 3.14159265358979323846
#define M_1_PI 0.318309886183790671538
#define EPSILON 0.0001

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define NUM_RAY_TYPES 2
#define MAX_PATH_DEPTH 5

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

// http://www.pcg-random.org/download.html
struct PCGRand {
	uint64_t state;
	// Just use stream 1
};

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

	float ior;
	float specular_transmission;
};

cbuffer MaterialParams : register(b1) {
	float4 basecolor_metallic;
	float4 spec_rough_spectint_aniso;
	float4 sheen_sheentint_clearc_ccgloss;
	float4 ior_spectrans;
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

float3 spherical_dir(float sin_theta, float cos_theta, float phi) {
	return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
	float f = n_f * pdf_f;
	float g = n_g * pdf_g;
	return (f * f) / (f * f + g * g);
}

float luminance(in const float3 c) {
	return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

float schlick_weight(float cos_theta) {
	return pow(saturate(1.f - cos_theta), 5.f);
}

float pow2(float x) {
	return x * x;
}

// Complete Fresnel Dielectric computation, for transmission at ior near 1
// they mention having issues with the Schlick approximation.
// eta_i: material on incident side's ior
// eta_t: material on transmitted side's ior
float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
	float g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
	if (g < 0.f) {
		return 1.f;
	}
	return 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i)
		* (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) / pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
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
	return M_1_PI * alpha_sqr / pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, float2 alpha) {
	return M_1_PI / (alpha.x * alpha.y
			* pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
	float a = alpha_g * alpha_g;
	float b = n_dot_o * n_dot_o;
	return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, float2 alpha) {
	return 1.f / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
}

// Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
float3 sample_lambertian_dir(in const float3 n, in const float3 v_x, in const float3 v_y, in const float2 s) {
	const float3 hemi_dir = normalize(cos_sample_hemisphere(s));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
float3 sample_gtr_1_h(in const float3 n, in const float3 v_x, in const float3 v_y, float alpha, in const float2 s) {
	float phi_h = 2.f * M_PI * s.x;
	float alpha_sqr = alpha * alpha;
	float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
	float cos_theta_h = sqrt(cos_theta_h_sqr);
	float sin_theta_h = 1.f - cos_theta_h_sqr;
	float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

float3 sample_gtr_2_h(in const float3 n, in const float3 v_x, in const float3 v_y, float alpha, in const float2 s) {
	float phi_h = 2.f * M_PI * s.x;
	float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
	float cos_theta_h = sqrt(cos_theta_h_sqr);
	float sin_theta_h = 1.f - cos_theta_h_sqr;
	float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
	return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

float3 sample_gtr_2_aniso_h(in const float3 n, in const float3 v_x, in const float3 v_y, in const float2 alpha, in const float2 s) {
	float x = 2.f * M_PI * s.x;
	float3 w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
	return normalize(w_h);
}

float lambertian_pdf(in const float3 w_i, in const float3 n) {
	float d = dot(w_i, n);
	if (d > 0.f) {
		return d * M_1_PI;
	}
	return 0.f;
}

float gtr_1_pdf(in const float3 w_o, in const float3 w_i, in const float3 n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_1(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float gtr_2_pdf(in const float3 w_o, in const float3 w_i, in const float3 n, float alpha) {
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2(cos_theta_h, alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float gtr_2_transmission_pdf(in const float3 w_o, in const float3 w_i, in const float3 n,
	float alpha, float ior)
{
	if (same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	bool entering = dot(w_o, n) > 0.f;
	float eta_o = entering ? 1.f : ior;
	float eta_i = entering ? ior : 1.f;
	float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
	float cos_theta_h = abs(dot(n, w_h));
	float i_dot_h = dot(w_i, w_h);
	float o_dot_h = dot(w_o, w_h);
	float d = gtr_2(cos_theta_h, alpha);
	float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
	return d * cos_theta_h * abs(dwh_dwi);
}

float gtr_2_aniso_pdf(in const float3 w_o, in const float3 w_i, in const float3 n,
	in const float3 v_x, in const float3 v_y, const float2 alpha)
{
	if (!same_hemisphere(w_o, w_i, n)) {
		return 0.f;
	}
	float3 w_h = normalize(w_i + w_o);
	float cos_theta_h = dot(n, w_h);
	float d = gtr_2_aniso(cos_theta_h, abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
	return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

float3 disney_diffuse(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float n_dot_o = abs(dot(w_o, n));
	float n_dot_i = abs(dot(w_i, n));
	float i_dot_h = dot(w_i, w_h);
	float fd90 = 0.5f + 2.f * mat.roughness * i_dot_h * i_dot_h;
	float fi = schlick_weight(n_dot_i);
	float fo = schlick_weight(n_dot_o);
	return mat.base_color * M_1_PI * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

float3 disney_microfacet_isotropic(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : float3(1, 1, 1);
	float3 spec = lerp(mat.specular * 0.08 * lerp(float3(1, 1, 1), tint, mat.specular_tint), mat.base_color, mat.metallic);

	float alpha = max(0.001, mat.roughness * mat.roughness);
	float d = gtr_2(dot(n, w_h), alpha);
	float3 f = lerp(spec, float3(1, 1, 1), schlick_weight(dot(w_i, w_h)));
	float g = smith_shadowing_ggx(dot(n, w_i), alpha) * smith_shadowing_ggx(dot(n, w_o), alpha);
	return d * f * g;
}

float3 disney_microfacet_transmission_isotropic(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i)
{
	float o_dot_n = dot(w_o, n);
	float i_dot_n = dot(w_i, n);
	if (o_dot_n == 0.f || i_dot_n == 0.f) {
		return 0.f;
	}
	bool entering = o_dot_n > 0.f;
	float eta_o = entering ? 1.f : mat.ior;
	float eta_i = entering ? mat.ior : 1.f;
	float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
	float3 spec = mat.specular_transmission * mat.base_color;

	float alpha = max(0.001, mat.roughness * mat.roughness);
	float d = gtr_2(abs(dot(n, w_h)), alpha);

	float f = fresnel_dielectric(abs(dot(w_i, n)), eta_o, eta_i);
	float g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) * smith_shadowing_ggx(abs(dot(n, w_o)), alpha);

	float i_dot_h = dot(w_i, w_h);
	float o_dot_h = dot(w_o, w_h);

	float c = abs(o_dot_h) / abs(dot(w_o, n)) * abs(i_dot_h) / abs(dot(w_i, n))
		* pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);

	return spec * c * (1.f - f) * g * d;
}

float3 disney_microfacet_anisotropic(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 v_x, in const float3 v_y)
{
	float3 w_h = normalize(w_i + w_o);
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
	in const float3 w_o, in const float3 w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
	float d = gtr_1(dot(n, w_h), alpha);
	float f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, n)));
	float g = smith_shadowing_ggx(dot(n, w_i), 0.25f) * smith_shadowing_ggx(dot(n, w_o), 0.25f);
	return 0.25 * mat.clearcoat * d * f * g;
}

float3 disney_sheen(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i)
{
	float3 w_h = normalize(w_i + w_o);
	float lum = luminance(mat.base_color);
	float3 tint = lum > 0.f ? mat.base_color / lum : float3(1, 1, 1);
	float3 sheen_color = lerp(float3(1, 1, 1), tint, mat.sheen_tint);
	float f = schlick_weight(dot(w_i, n));
	return f * mat.sheen * sheen_color;
}

float3 disney_brdf(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 v_x, in const float3 v_y)
{
	if (!same_hemisphere(w_o, w_i, n)) {
		if (mat.specular_transmission > 0.f) {
			float3 spec_trans = disney_microfacet_transmission_isotropic(mat, n, w_o, w_i);
			return spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
		}
		return 0.f;
	}

	float coat = disney_clear_coat(mat, n, w_o, w_i);
	float3 sheen = disney_sheen(mat, n, w_o, w_i);
	float3 diffuse = disney_diffuse(mat, n, w_o, w_i);
	float3 gloss;
	if (mat.anisotropy == 0.f) {
		gloss = disney_microfacet_isotropic(mat, n, w_o, w_i);
	} else {
		gloss = disney_microfacet_anisotropic(mat, n, w_o, w_i, v_x, v_y);
	}
	return (diffuse + sheen) * (1.f - mat.metallic) * (1.f - mat.specular_transmission) + gloss + coat;
}

float disney_pdf(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 w_i, in const float3 v_x, in const float3 v_y)
{
	float alpha = max(0.001, mat.roughness * mat.roughness);
	float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
	float2 alpha_aniso = float2(max(0.001, alpha / aspect), max(0.001, alpha * aspect));

	float clearcoat_alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);

	float diffuse = lambertian_pdf(w_i, n);
	float clear_coat = gtr_1_pdf(w_o, w_i, n, clearcoat_alpha);

	float n_comp = 3.f;
	float microfacet;
	float microfacet_transmission = 0.f;
	if (mat.anisotropy == 0.f) {
		microfacet = gtr_2_pdf(w_o, w_i, n, alpha);
	} else {
		microfacet = gtr_2_aniso_pdf(w_o, w_i, n, v_x, v_y, alpha_aniso);
	}
	if (mat.specular_transmission > 0.f) {
		n_comp = 4.f;
		microfacet_transmission = gtr_2_transmission_pdf(w_o, w_i, n, alpha, mat.ior);
	}
	return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
}

/* Sample a component of the Disney BRDF, returns the sampled BRDF color,
 * ray reflection direction (w_i) and sample PDF.
 */
float3 sample_disney_brdf(in const DisneyMaterial mat, in const float3 n,
	in const float3 w_o, in const float3 v_x, in const float3 v_y, inout PCGRand rng,
	out float3 w_i, out float pdf)
{
	int component = abs(pcg32_random(rng));
	if (mat.specular_transmission == 0.f) {
		component = component % 3;
	} else {
		component = component % 4;
	}

	float2 samples = float2(pcg32_randomf(rng), pcg32_randomf(rng));
	if (component == 0) {
		// Sample diffuse component
		w_i = sample_lambertian_dir(n, v_x, v_y, samples);
	} else if (component == 1) {
		float3 w_h;
		float alpha = max(0.001, mat.roughness * mat.roughness);
		if (mat.anisotropy == 0.f) {
			w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
		} else {
			float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
			float2 alpha_aniso = float2(max(0.001, alpha / aspect), max(0.001, alpha * aspect));
			w_h = sample_gtr_2_aniso_h(n, v_x, v_y, alpha_aniso, samples);
		}
		w_i = reflect(-w_o, w_h);

		// Invalid reflection, terminate ray
		if (!same_hemisphere(w_o, w_i, n)) {
			pdf = 0.f;
			w_i = 0.f;
			return 0.f;
		}
	} else if (component == 2) {
		// Sample clear coat component
		float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
		float3 w_h = sample_gtr_1_h(n, v_x, v_y, alpha, samples);
		w_i = reflect(-w_o, w_h);

		// Invalid reflection, terminate ray
		if (!same_hemisphere(w_o, w_i, n)) {
			pdf = 0.f;
			w_i = 0.f;
			return 0.f;
		}
	} else {
		// Sample microfacet transmission component
		float alpha = max(0.001, mat.roughness * mat.roughness);
		float3 w_h = sample_gtr_2_h(n, v_x, v_y, alpha, samples);
		if (dot(w_o, w_h) < 0.f) {
			w_h = -w_h;
		}
		bool entering = dot(w_o, n) > 0.f;
		w_i = refract(-w_o, w_h, entering ? 1.f / mat.ior : mat.ior);

		// Invalid refraction, terminate ray
		if (all(w_i == float3(0.f, 0.f, 0.f))) {
			pdf = 0.f;
			return 0.f;
		}
	}
	pdf = disney_pdf(mat, n, w_o, w_i, v_x, v_y);
	return disney_brdf(mat, n, w_o, w_i, v_x, v_y);
}

// Quad-shaped light source
struct QuadLight {
	float4 emission;
	float4 position;
	float4 normal;
	// x and y vectors spanning the quad, with
	// the half-width and height in the w component
	float4 v_x;
	float4 v_y;
};

float3 sample_quad_light_position(in const QuadLight light, float2 samples) {
	return samples.x * light.v_x.xyz * light.v_x.w
		+ samples.y * light.v_y.xyz * light.v_y.w + light.position.xyz;
}

/* Compute the PDF of sampling the sampled point p light with the ray specified by orig and dir,
 * assuming the light is not occluded
 */
float quad_light_pdf(in const QuadLight light, in const float3 p, in const float3 orig, in const float3 dir) {
	float surface_area = light.v_x.w * light.v_y.w;
	float3 to_pt = p - dir;
	float dist_sqr = dot(to_pt, to_pt);
	float n_dot_w = dot(light.normal.xyz, -dir);
	if (n_dot_w < EPSILON) {
		return 0.f;
	}
	return dist_sqr / (n_dot_w * surface_area);
}

bool quad_intersect(in const QuadLight light, in const float3 orig, in const float3 dir,
	out float t, out float3 light_pos)
{
	float denom = dot(dir, light.normal.xyz);
	if (denom >= EPSILON) {
		t = dot(light.position.xyz - orig, light.normal.xyz) / denom;
		if (t < 0.f) {
			return false;
		}

		// It's a finite plane so now see if the hit point is actually inside the plane
		light_pos = orig + dir * t;
		float3 hit_v = light_pos - light.position.xyz;
		if (abs(dot(hit_v, light.v_x.xyz)) < light.v_x.w && abs(dot(hit_v, light.v_y.xyz)) < light.v_y.w) {
			return true;
		}
	}
	return false;
}

float3 sample_direct_light(in const DisneyMaterial mat, in const float3 hit_p, in const float3 n,
	in const float3 v_x, in const float3 v_y, in const float3 w_o, inout PCGRand rng)
{
	float3 illum = 0.f;

	QuadLight light;
	light.emission = 5.f;
	light.normal.xyz = normalize(float3(0.5, -0.8, -0.5));
	light.position.xyz = 10.f * -light.normal.xyz;
	// TODO: This would be input from the scene telling us how the light is placed
	// For now we don't care
	ortho_basis(light.v_x.xyz, light.v_y.xyz, light.normal.xyz);
	light.v_x.w = 5.f;
	light.v_y.w = 5.f;

	OcclusionHitInfo shadow_hit;
	RayDesc shadow_ray;
	shadow_ray.Origin = hit_p;
	shadow_ray.TMin = EPSILON;

	// Sample the light to compute an incident light ray to this point
	{
		float3 light_pos = sample_quad_light_position(light, float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		float3 light_dir = light_pos - hit_p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
		// TODO: Maybe should check if same hemisphere here?
		float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

		shadow_ray.Direction = light_dir;
		shadow_ray.TMax = light_dist;
		TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
				OCCLUSION_RAY, NUM_RAY_TYPES, OCCLUSION_RAY, shadow_ray, shadow_hit);

		if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_hit.hit == 0) {
			float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
			float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
			illum = bsdf * light.emission.rgb * abs(dot(light_dir, n)) * w / light_pdf;
		}
	}

	// Sample the BRDF to compute a light sample as well
	{
		float3 w_i;
		float bsdf_pdf;
		float3 bsdf = sample_disney_brdf(mat, n, w_o, v_x, v_y, rng, w_i, bsdf_pdf);
		
		float light_dist;
		float3 light_pos;
		if (all(bsdf > 0.f) && bsdf_pdf >= EPSILON && quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
			float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
			if (light_pdf >= EPSILON) {
				float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);

				shadow_ray.Direction = w_i;
				shadow_ray.TMax = light_dist;
				TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
						OCCLUSION_RAY, NUM_RAY_TYPES, OCCLUSION_RAY, shadow_ray, shadow_hit);
				if (shadow_hit.hit == 0) {
					illum += bsdf * light.emission.rgb * abs(dot(w_i, n)) * w / bsdf_pdf;
				}
			}
		}
	}
	return illum;
}


[shader("raygeneration")] 
void RayGen() {
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

	mat.ior = ior_spectrans.r;
	mat.specular_transmission = ior_spectrans.g;

	int bounce = 0;
	float3 illum = float3(0, 0, 0);
	float3 path_throughput = float3(1, 1, 1);
	do {
		HitInfo payload;
		payload.color_dist = float4(0, 0, 0, -1);
		TraceRay(scene, 0, 0xff, PRIMARY_RAY, NUM_RAY_TYPES, PRIMARY_RAY, ray, payload);

		// If we hit nothing, include the scene background color from the miss shader
		if (payload.color_dist.w <= 0) {
			illum += path_throughput * payload.color_dist.rgb;
			break;
		}

		const float3 w_o = -ray.Direction;
		const float3 hit_p = ray.Origin + payload.color_dist.w * ray.Direction;
		float3 v_x, v_y;
		float3 v_z = normalize(payload.normal.xyz);
		ortho_basis(v_x, v_y, v_z);

		illum += path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o, rng);

		float3 w_i;
		float pdf;
		float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
		if (pdf < EPSILON || all(bsdf == 0.f)) {
			break;
		}
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

		if (path_throughput.x < EPSILON && path_throughput.y < EPSILON && path_throughput.z < EPSILON) {
			break;
		}

		ray.Origin = hit_p;
		ray.Direction = w_i;
		ray.TMin = EPSILON;
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
	payload.color_dist.w = -1.f;

	float3 dir = WorldRayDirection();
	float u = (1.f + atan2(dir.x, -dir.z) * M_1_PI) * 0.5f;
	float v = acos(dir.y) * M_1_PI;

	int check_x = u * 10.f;
	int check_y = v * 10.f;

	if (dir.y > -0.1 && (check_x + check_y) % 2 == 0) {
		payload.color_dist.rgb = 0.5f;// * (1.f + normalize(dir)) * 0.5f;
	} else {
		payload.color_dist.rgb = 0.1f;
	}
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
