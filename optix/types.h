#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

#define EPSILON 0.0001

#define PRIMARY_RAY 0
#define OCCLUSION_RAY 1
#define NUM_RAY_TYPES 2
#define MAX_PATH_DEPTH 5

#ifdef __CUDA_ARCH__
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
#endif

