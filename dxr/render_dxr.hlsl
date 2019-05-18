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

		const float3 hit_p = ray.Origin + payload.color_dist.w * ray.Direction;
		float3 v_x, v_y;
		float3 v_z = normalize(payload.normal.xyz);
		ortho_basis(v_x, v_y, v_z);

		payload.color_dist.rgb = float3(0.9, 0.9, 0.9);
		const float3 bsdf = payload.color_dist.rgb * M_1_PI;

		// Direct light sampling.
		// Note: we just treat the camera position as being the location of a point light
		const float3 w_o = -ray.Direction;

		//float3 light_dir = cam_pos.xyz - hit_p;
		//float light_dist = length(light_dir);
		float3 light_dir = float3(-0.5, 0.8, 0.5);
		//float light_dist = 1.0;
		light_dir = normalize(light_dir);

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
		// TODO: Cosine weighted hemisphere sampling
		const float theta = sqrt(pcg32_randomf(rng));
		const float phi = 2.0f * M_PI * pcg32_randomf(rng);

		const float x = cos(phi) * theta;
		const float y = sin(phi) * theta;
		const float z = sqrt(1.0 - theta * theta);

		float3 w_i;
		w_i.x = x * v_x.x + y * v_y.x + z * v_z.x;
		w_i.y = x * v_x.y + y * v_y.y + z * v_z.y;
		w_i.z = x * v_x.z + y * v_y.z + z * v_z.z;
		w_i = normalize(w_i);

		// Update path throughput and continue the ray
		// TODO: This is just a hard-coded Lambertian BRDF,
		// using the object's normal color as its albedo
		float pdf = abs(dot(w_i, v_z)) * M_1_PI;
		if (pdf == 0.0) {
			break;
		}
		// Note: same as just multiplying my M_PI b/c the cancellation,
		// but left like this b/c I'll swap to Disney BRDF soon-ish
		path_throughput *= bsdf * abs(dot(w_i, v_z)) / pdf;
		if (path_throughput.x == 0 && path_throughput.y == 0 && path_throughput.z == 0) {
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
	float3 c = (n + float3(1, 1, 1)) * 0.5;
	payload.color_dist = float4(c, RayTCurrent());
	payload.normal = float4(n, 0);
}

[shader("closesthit")]
void AOHit(inout OcclusionHitInfo occlusion, Attributes attrib) {
	occlusion.hit = 1;
}
