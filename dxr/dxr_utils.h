#pragma once

#include "dx12_utils.h"

// Utilities for DXR ease of use

bool dxr_available(Microsoft::WRL::ComPtr<ID3D12Device5> &device);
