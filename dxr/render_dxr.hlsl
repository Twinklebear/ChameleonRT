// https://developer.nvidia.com/rtx/raytracing/dxr/DX12-Raytracing-tutorial-Part-2
struct HitInfo {
	float4 color_dist;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
	float2 bary;
};

// Raytracing output texture, accessed as a UAV
RWTexture2D<float4> output : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure scene : register(t0);

[shader("raygeneration")] 
void RayGen() {
	// Initialize the ray payload
	HitInfo payload;
	payload.color_dist = float4(0, 0, 1, 1);

	// Get the location within the dispatched 2D grid of work items
	// (often maps to pixels, so this could represent a pixel coordinate).
	uint2 pixel = DispatchRaysIndex().xy;
	float2 dims = float2(DispatchRaysDimensions().xy);

	// The example traces an orthographic view essentially (or NDC "projection")
	float2 dir = (((pixel.xy + 0.5f) / dims.xy) * 2.f - 1.f);
	RayDesc ray;
	ray.Origin = float3(dir.x, -dir.y, 1);
	ray.Direction = float3(0, 0, -1);
	ray.TMin = 0;
	ray.TMax = 1e20f;

	TraceRay(scene, 0, 0xff, 0, 0, 0, ray, payload);

	output[pixel] = float4(payload.color_dist.rgb, 1.f);
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload) {
	payload.color_dist = float4(0.0f, 0.2f, 0.4f, -1.f);
}

[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) {
	float3 barycoords = float3(1.f - attrib.bary.x - attrib.bary.y,
		attrib.bary.x, attrib.bary.y);
	payload.color_dist = float4(barycoords, RayTCurrent());
}

