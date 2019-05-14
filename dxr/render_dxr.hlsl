#define M_PI 3.14159265358979323846

struct HitInfo {
	float4 color_dist;
	float4 normal;
};

struct AOHitInfo {
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
	// Ignoring the stream selection and just using 1 always
};

PCGRand get_rng() {
	PCGRand rng;
	// TODO: These might only be available in the raygen program
	uint2 pixel = DispatchRaysIndex().xy;
	rng.state = pixel.x + pixel.y * DispatchRaysDimensions().x;
	return rng;
}

uint pcg32_random(inout PCGRand rng) {
	uint64_t oldstate = rng.state;
	rng.state = oldstate * 6364136223846793005ULL;
	// Calculate output function (XSH RR), uses old state for max ILP
	uint xorshifted = uint(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint rot = uint(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float pcg32_randomf(inout PCGRand rng) {
	return ldexp(pcg32_random(rng), -32);
}

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> output : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

// View params buffer
cbuffer ViewParams : register(b0) {
	float4 cam_pos;
	float4 cam_du;
	float4 cam_dv;
	float4 cam_dir_top_left;
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

	uint2 pixel = DispatchRaysIndex().xy;
	float2 dims = float2(DispatchRaysDimensions().xy);

	float2 d = pixel / dims;
	RayDesc ray;
	ray.Origin = cam_pos.xyz;
	ray.Direction = normalize(d.x * cam_du.xyz + d.y * cam_dv.xyz + cam_dir_top_left.xyz);
	ray.TMin = 0;
	ray.TMax = 1e20f;

	HitInfo payload;
	payload.color_dist = float4(0, 0, 0, -1);
	TraceRay(scene, 0, 0xff, PRIMARY_RAY, NUM_RAY_TYPES, PRIMARY_RAY, ray, payload);

	float occlusion = 1.f;
	if (payload.color_dist.w > 0) {
		const float3 hit_p = ray.Origin + payload.color_dist.w * ray.Direction;
		float3 v_x, v_y;
		float3 v_z = payload.normal.xyz;
		ortho_basis(v_x, v_y, v_z);
		PCGRand rng = get_rng();

		int ao_samples = 4;
		int num_occluded = 0;
		for (int i = 0; i < ao_samples; ++i) {
			for (int j = 0; j < ao_samples; ++j) {
				// Sample the hemisphere
				const float theta = sqrt(pcg32_randomf(rng));
				const float phi = 2.0f * M_PI * pcg32_randomf(rng);

				const float x = cos(phi) * theta;
				const float y = sin(phi) * theta;
				const float z = sqrt(1.0 - theta * theta);
				
				RayDesc ao_ray;
				ao_ray.Origin = hit_p;
				ao_ray.Direction.x = x * v_x.x + y * v_y.x + z * v_z.x;
				ao_ray.Direction.y = x * v_x.y + y * v_y.y + z * v_z.y;
				ao_ray.Direction.z = x * v_x.z + y * v_y.z + z * v_z.z;
				ao_ray.TMin = 0.0001;
				ao_ray.TMax = 1e20f;
		
				AOHitInfo aohit;
				TraceRay(scene, 0, 0xff, OCCLUSION_RAY, NUM_RAY_TYPES, OCCLUSION_RAY, ao_ray, aohit);
				if (aohit.hit == 1) {
					++num_occluded;
				}

			}
		}
		float total_ao_samples = ao_samples * ao_samples;
		occlusion = (total_ao_samples - num_occluded) / total_ao_samples;
	}
	payload.color_dist.rbg *= occlusion;
	output[pixel] = float4(linear_to_srgb(payload.color_dist.r),
		linear_to_srgb(payload.color_dist.g),
		linear_to_srgb(payload.color_dist.b), 1.f);
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload) {
	payload.color_dist = float4(0, 0, 0, 0);
}

[shader("miss")]
void AOMiss(inout AOHitInfo aohit : SV_RayPayload) {
	aohit.hit = 0;
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
void AOHit(inout AOHitInfo aohit, Attributes attrib) {
	aohit.hit = 1;
}
