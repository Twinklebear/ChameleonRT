#pragma once

#include <string>

enum DTYPE {
    INT_8,
    UINT_8,
    INT_16,
    UINT_16,
    INT_32,
    UINT_32,
    FLOAT_32,
    FLOAT_64,
    VEC2_I8,
    VEC2_U8,
    VEC2_I16,
    VEC2_U16,
    VEC2_I32,
    VEC2_U32,
    VEC2_F32,
    VEC2_F64,
    VEC3_I8,
    VEC3_U8,
    VEC3_I16,
    VEC3_U16,
    VEC3_I32,
    VEC3_U32,
    VEC3_F32,
    VEC3_F64,
    VEC4_I8,
    VEC4_U8,
    VEC4_I16,
    VEC4_U16,
    VEC4_I32,
    VEC4_U32,
    VEC4_F32,
    VEC4_F64,
    MAT2_I8,
    MAT2_U8,
    MAT2_I16,
    MAT2_U16,
    MAT2_I32,
    MAT2_U32,
    MAT2_F32,
    MAT2_F64,
    MAT3_I8,
    MAT3_U8,
    MAT3_I16,
    MAT3_U16,
    MAT3_I32,
    MAT3_U32,
    MAT3_F32,
    MAT3_F64,
    MAT4_I8,
    MAT4_U8,
    MAT4_I16,
    MAT4_U16,
    MAT4_I32,
    MAT4_U32,
    MAT4_F32,
    MAT4_F64,
};

std::string print_primitive_mode(int mode);

std::string print_data_type(DTYPE type);

DTYPE parse_dtype(const std::string &str);

size_t gltf_base_stride(int type, int component_type);

DTYPE gltf_type_to_dtype(int type, int component_type);

size_t dtype_stride(DTYPE type);

size_t dtype_components(DTYPE type);

