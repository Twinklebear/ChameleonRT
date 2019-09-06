#pragma once

#include <string>

enum DTYPE {
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    FLOAT,
    DOUBLE,
    VEC2_I8,
    VEC2_U8,
    VEC2_I16,
    VEC2_U16,
    VEC2_I32,
    VEC2_U32,
    VEC2_FLOAT,
    VEC2_DOUBLE,
    VEC3_I8,
    VEC3_U8,
    VEC3_I16,
    VEC3_U16,
    VEC3_I32,
    VEC3_U32,
    VEC3_FLOAT,
    VEC3_DOUBLE,
    VEC4_I8,
    VEC4_U8,
    VEC4_I16,
    VEC4_U16,
    VEC4_I32,
    VEC4_U32,
    VEC4_FLOAT,
    VEC4_DOUBLE,
    MAT2_I8,
    MAT2_U8,
    MAT2_I16,
    MAT2_U16,
    MAT2_I32,
    MAT2_U32,
    MAT2_FLOAT,
    MAT2_DOUBLE,
    MAT3_I8,
    MAT3_U8,
    MAT3_I16,
    MAT3_U16,
    MAT3_I32,
    MAT3_U32,
    MAT3_FLOAT,
    MAT3_DOUBLE,
    MAT4_I8,
    MAT4_U8,
    MAT4_I16,
    MAT4_U16,
    MAT4_I32,
    MAT4_U32,
    MAT4_FLOAT,
    MAT4_DOUBLE,
};

std::string print_primitive_mode(int mode);

std::string print_gltf_data_type(int ty);

std::string print_gltf_component_type(int ty);

size_t gltf_base_stride(int type, int component_type);

DTYPE gltf_type_to_dtype(int type, int component_type);

size_t dtype_stride(DTYPE type);

size_t dtype_components(DTYPE type);

