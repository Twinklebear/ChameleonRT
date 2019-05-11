#include "dxr_utils.h"

using Microsoft::WRL::ComPtr;

bool dxr_available(ComPtr<ID3D12Device5> &device) {
	D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_data = { 0 };
	CHECK_ERR(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
		&feature_data, sizeof(feature_data)));

	return feature_data.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}