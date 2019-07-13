#include "util.hlsl"
#include "pcg_rng.hlsl"
#include "disney_bsdf.hlsl"
#include "lights.hlsl"

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

struct MaterialParams {
	// TODO is this packing into the float4 needed? it's not going through a cbuffer now
	float4 basecolor_metallic;
	float4 spec_rough_spectint_aniso;
	float4 sheen_sheentint_clearc_ccgloss;
	float4 ior_spectrans;
	int4 tex_ids;
};

// Instance ID = Material ID
StructuredBuffer<MaterialParams> material_params : register(t1);

Texture2D textures[] : register(t2);
SamplerState tex_sampler : register(s0);

void unpack_material(inout DisneyMaterial mat, uint id, float2 uv_coords) {
	MaterialParams p = material_params[id];
	if (p.tex_ids.x < 0) {
		mat.base_color = p.basecolor_metallic.rgb;
	} else {
		mat.base_color = textures[NonUniformResourceIndex(p.tex_ids.x)]
			.SampleLevel(tex_sampler, uv_coords, 0).rgb;
	}

	mat.metallic = p.basecolor_metallic.a;
	mat.specular = p.spec_rough_spectint_aniso.r;
	mat.roughness = p.spec_rough_spectint_aniso.g;
	mat.specular_tint = p.spec_rough_spectint_aniso.b;
	mat.anisotropy = p.spec_rough_spectint_aniso.a;
	mat.sheen = p.sheen_sheentint_clearc_ccgloss.r;
	mat.sheen_tint = p.sheen_sheentint_clearc_ccgloss.g;
	mat.clearcoat = p.sheen_sheentint_clearc_ccgloss.b;
	mat.clearcoat_gloss = p.sheen_sheentint_clearc_ccgloss.a;
	mat.ior = p.ior_spectrans.r;
	mat.specular_transmission = p.ior_spectrans.g;
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
		float3 light_pos = sample_quad_light_position(light,
			float2(pcg32_randomf(rng), pcg32_randomf(rng)));
		float3 light_dir = light_pos - hit_p;
		float light_dist = length(light_dir);
		light_dir = normalize(light_dir);

		float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
		float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

		shadow_ray.Direction = light_dir;
		shadow_ray.TMax = light_dist;
		TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
				OCCLUSION_RAY, 0, OCCLUSION_RAY, shadow_ray, shadow_hit);

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
		if (any(bsdf > 0.f) && bsdf_pdf >= EPSILON
			&& quad_intersect(light, hit_p, w_i, light_dist, light_pos))
		{
			float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
			if (light_pdf >= EPSILON) {
				float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);

				shadow_ray.Direction = w_i;
				shadow_ray.TMax = light_dist;
				TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff,
						OCCLUSION_RAY, 0, OCCLUSION_RAY, shadow_ray, shadow_hit);
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
	PCGRand rng = get_rng(frame_id);
	float2 d = (pixel + float2(pcg32_randomf(rng), pcg32_randomf(rng))) / dims;

	RayDesc ray;
	ray.Origin = cam_pos.xyz;
	ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
	ray.TMin = 0;
	ray.TMax = 1e20f;

	DisneyMaterial mat;

	int bounce = 0;
	float3 illum = float3(0, 0, 0);
	float3 path_throughput = float3(1, 1, 1);
	do {
		HitInfo payload;
		payload.color_dist = float4(0, 0, 0, -1);
		TraceRay(scene, 0, 0xff, PRIMARY_RAY, 0, PRIMARY_RAY, ray, payload);

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

		unpack_material(mat, uint(payload.normal.w), payload.color_dist.rg);
		illum += path_throughput * sample_direct_light(mat, hit_p, v_z, v_x, v_y, w_o, rng);

		float3 w_i;
		float pdf;
		float3 bsdf = sample_disney_brdf(mat, v_z, w_o, v_x, v_y, rng, w_i, pdf);
		if (pdf < EPSILON || all(bsdf == 0.f)) {
			break;
		}
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;

		if (all(path_throughput < EPSILON)) {
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

// Per-mesh parameters for the closest hit
StructuredBuffer<float3> vertices : register(t0, space1);
StructuredBuffer<uint3> indices : register(t1, space1);
StructuredBuffer<float2> uvs : register(t2, space1);

cbuffer MeshData : register(b0, space1) {
	uint num_uvs;
}

[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) {
	uint3 idx = indices[PrimitiveIndex()];
	float3 va = vertices[idx.x];
	float3 vb = vertices[idx.y];
	float3 vc = vertices[idx.z];
	float3 n = normalize(cross(vb - va, vc - va));

	float2 uv = float2(0, 0);
	if (num_uvs > 0) {
		float2 uva = uvs[idx.x];
		float2 uvb = uvs[idx.y];
		float2 uvc = uvs[idx.z];
		uv = (1.f - attrib.bary.x - attrib.bary.y) * uva
			+ attrib.bary.x * uvb + attrib.bary.y * uvc;
	}

	payload.color_dist = float4(uv, 0, RayTCurrent());
	float3x3 inv_transp = float3x3(WorldToObject4x3()[0], WorldToObject4x3()[1], WorldToObject4x3()[2]);
	payload.normal = float4(normalize(mul(inv_transp, n)), InstanceID());
}

[shader("closesthit")]
void OcclusionHit(inout OcclusionHitInfo occlusion, Attributes attrib) {
	occlusion.hit = 1;
}
