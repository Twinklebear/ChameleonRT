#include "gltf_types.h"
#include "tiny_gltf.h"

std::string print_primitive_mode(int mode)
{
    if (mode == TINYGLTF_MODE_POINTS) {
        return "POINTS";
    } else if (mode == TINYGLTF_MODE_LINE) {
        return "LINE";
    } else if (mode == TINYGLTF_MODE_LINE_LOOP) {
        return "LINE_LOOP";
    } else if (mode == TINYGLTF_MODE_TRIANGLES) {
        return "TRIANGLES";
    } else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
        return "TRIANGLE_FAN";
    } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
        return "TRIANGLE_STRIP";
    }
    return "UNKNOWN MODE";
}

std::string print_gltf_data_type(int ty)
{
    if (ty == TINYGLTF_TYPE_SCALAR) {
        return "SCALAR";
    } else if (ty == TINYGLTF_TYPE_VECTOR) {
        // TODO: Does "vector" just refer to a variable sized vector? When would this be
        // encountered? How to know the size?
        return "VECTOR";
    } else if (ty == TINYGLTF_TYPE_VEC2) {
        return "VEC2";
    } else if (ty == TINYGLTF_TYPE_VEC3) {
        return "VEC3";
    } else if (ty == TINYGLTF_TYPE_VEC4) {
        return "VEC4";
    } else if (ty == TINYGLTF_TYPE_MATRIX) {
        // TODO: Does "vector" just refer to a variable sized matrix? When would this be
        // encountered? How to know the size?
        return "MATRIX";
    } else if (ty == TINYGLTF_TYPE_MAT2) {
        return "MAT2";
    } else if (ty == TINYGLTF_TYPE_MAT3) {
        return "MAT3";
    } else if (ty == TINYGLTF_TYPE_MAT4) {
        return "MAT4";
    }
    return "UNKNOWN DATA TYPE";
}

std::string print_gltf_component_type(int ty)
{
    if (ty == TINYGLTF_COMPONENT_TYPE_BYTE) {
        return "BYTE";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        return "UNSIGNED_BYTE";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_SHORT) {
        return "SHORT";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        return "UNSIGNED_SHORT";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_INT) {
        return "INT";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        return "UNSIGNED_INT";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return "FLOAT";
    } else if (ty == TINYGLTF_COMPONENT_TYPE_DOUBLE) {
        return "DOUBLE";
    }
    return "UNKNOWN GLTF COMPONENT TYPE";
}

size_t gltf_base_stride(int type, int component_type)
{
    return dtype_stride(gltf_type_to_dtype(type, component_type));
}

DTYPE gltf_type_to_dtype(int type, int component_type)
{
    switch (type) {
    case TINYGLTF_TYPE_SCALAR:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return INT8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return UINT8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return INT16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return UINT16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return INT32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return UINT32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_VEC2:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return VEC2_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return VEC2_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return VEC2_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return VEC2_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return VEC2_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return VEC2_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return VEC2_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC2_DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_VEC3:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return VEC3_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return VEC3_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return VEC3_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return VEC3_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return VEC3_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return VEC3_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return VEC3_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC3_DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_VEC4:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return VEC4_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return VEC4_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return VEC4_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return VEC4_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return VEC4_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return VEC4_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return VEC4_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC4_DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_MAT2:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return MAT2_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return MAT2_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return MAT2_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return MAT2_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return MAT2_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return MAT2_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return MAT2_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT2_DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_MAT3:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return MAT3_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return MAT3_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return MAT3_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return MAT3_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return MAT3_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return MAT3_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return MAT3_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT3_DOUBLE;
        default:
            break;
        };
    case TINYGLTF_TYPE_MAT4:
        switch (component_type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return MAT4_I8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return MAT4_U8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return MAT4_I16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return MAT4_U16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return MAT4_I32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return MAT4_U32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return MAT4_FLOAT;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT4_DOUBLE;
        default:
            break;
        };
    default:
        break;
    }
    throw std::runtime_error("Unrecognized type/component type pair");
}

size_t dtype_stride(DTYPE type)
{
    switch (type) {
    case INT8:
    case UINT8:
    case VEC2_I8:
    case VEC2_U8:
    case VEC3_I8:
    case VEC3_U8:
    case VEC4_I8:
    case VEC4_U8:
    case MAT2_I8:
    case MAT2_U8:
    case MAT3_I8:
    case MAT3_U8:
    case MAT4_I8:
    case MAT4_U8:
        return dtype_components(type);
    case INT16:
    case UINT16:
    case VEC2_I16:
    case VEC2_U16:
    case VEC3_I16:
    case VEC3_U16:
    case VEC4_I16:
    case VEC4_U16:
    case MAT2_I16:
    case MAT2_U16:
    case MAT3_I16:
    case MAT3_U16:
    case MAT4_I16:
    case MAT4_U16:
        return dtype_components(type) * 2;
    case INT32:
    case UINT32:
    case VEC2_I32:
    case VEC2_U32:
    case VEC3_I32:
    case VEC3_U32:
    case VEC4_I32:
    case VEC4_U32:
    case MAT2_I32:
    case MAT2_U32:
    case MAT3_I32:
    case MAT3_U32:
    case MAT4_I32:
    case MAT4_U32:
    case FLOAT:
    case VEC2_FLOAT:
    case VEC3_FLOAT:
    case VEC4_FLOAT:
    case MAT2_FLOAT:
    case MAT3_FLOAT:
    case MAT4_FLOAT:
        return dtype_components(type) * 4;
    case DOUBLE:
    case VEC2_DOUBLE:
    case VEC3_DOUBLE:
    case VEC4_DOUBLE:
    case MAT2_DOUBLE:
    case MAT3_DOUBLE:
    case MAT4_DOUBLE:
        return dtype_components(type) * 8;
    default:
        break
    }
    throw std::runtime_error("UNKOWN DATATYPE");
}

size_t dtype_components(DTYPE type)
{
    switch (type) {
    case INT8:
    case UINT8:
    case INT16:
    case UINT16:
    case INT32:
    case UINT32:
    case FLOAT:
    case DOUBLE:
        return 1;
    case VEC2_I8:
    case VEC2_U8:
    case VEC2_I16:
    case VEC2_U16:
    case VEC2_I32:
    case VEC2_U32:
    case VEC2_FLOAT:
    case VEC2_DOUBLE:
        return 2;
    case VEC3_I8:
    case VEC3_U8:
    case VEC3_I16:
    case VEC3_U16:
    case VEC3_I32:
    case VEC3_U32:
    case VEC3_FLOAT:
    case VEC3_DOUBLE:
        return 3;
    case VEC4_I8:
    case VEC4_U8:
    case VEC4_I16:
    case VEC4_U16:
    case VEC4_I32:
    case VEC4_U32:
    case VEC4_FLOAT:
    case VEC4_DOUBLE:
    case MAT2_I8:
    case MAT2_U8:
    case MAT2_I16:
    case MAT2_U16:
    case MAT2_I32:
    case MAT2_U32:
    case MAT2_FLOAT:
    case MAT2_DOUBLE:
        return 4;
    case MAT3_I8:
    case MAT3_U8:
    case MAT3_I16:
    case MAT3_U16:
    case MAT3_I32:
    case MAT3_U32:
    case MAT3_FLOAT:
    case MAT3_DOUBLE:
        return 9;
    case MAT4_I8:
    case MAT4_U8:
    case MAT4_I16:
    case MAT4_U16:
    case MAT4_I32:
    case MAT4_U32:
    case MAT4_FLOAT:
    case MAT4_DOUBLE:
        return 16;
    default:
        break;
    }
    throw std::runtime_error("Invalid data type");
}

