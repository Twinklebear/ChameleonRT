#pragma once

#define CHECK_OPTIX(FN) \
	{ \
		auto fn_err = FN; \
		if (fn_err != OPTIX_SUCCESS) { \
			std::cout << #FN << " failed due to " \
				<< optixGetErrorName(fn_err) << ": " << optixGetErrorString(fn_err) \
				<< std::endl << std::flush; \
			throw std::runtime_error(#FN); \
		}\
	}

#define CHECK_CUDA(FN) \
	{ \
		auto fn_err = FN; \
		if (fn_err != cudaSuccess) { \
			std::cout << #FN << " failed due to " \
				<< cudaGetErrorName(fn_err) << ": " << cudaGetErrorString(fn_err) \
				<< std::endl << std::flush; \
			throw std::runtime_error(#FN); \
		}\
	}

