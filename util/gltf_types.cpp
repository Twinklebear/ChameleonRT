#include "gltf_types.h"
#include <iostream>
#include <string>
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

std::string print_data_type(DTYPE type)
{
    switch (type) {
    case DTYPE::INT8:
        return "INT8";
    case DTYPE::UINT8:
        return "UINT8";
    case DTYPE::INT16:
        return "INT16";
    case DTYPE::UINT16:
        return "UINT16";
    case DTYPE::INT32:
        return "INT32";
    case DTYPE::UINT32:
        return "UINT32";
    case DTYPE::FLOAT:
        return "FLOAT";
    case DOUBLE:
        return "DOUBLE";
    case VEC2_I8:
        return "VEC2_I8";
    case VEC2_U8:
        return "VEC2_U8";
    case VEC2_I16:
        return "VEC2_I16";
    case VEC2_U16:
        return "VEC2_U16";
    case VEC2_I32:
        return "VEC2_I32";
    case VEC2_U32:
        return "VEC2_U32";
    case VEC2_FLOAT:
        return "VEC2_FLOAT";
    case VEC2_DOUBLE:
        return "VEC2_DOUBLE";
    case VEC3_I8:
        return "VEC3_I8";
    case VEC3_U8:
        return "VEC3_U8";
    case VEC3_I16:
        return "VEC3_I16";
    case VEC3_U16:
        return "VEC3_U16";
    case VEC3_I32:
        return "VEC3_I32";
    case VEC3_U32:
        return "VEC3_U32";
    case VEC3_FLOAT:
        return "VEC3_FLOAT";
    case VEC3_DOUBLE:
        return "VEC3_DOUBLE";
    case VEC4_I8:
        return "VEC4_I8";
    case VEC4_U8:
        return "VEC4_U8";
    case VEC4_I16:
        return "VEC4_I16";
    case VEC4_U16:
        return "VEC4_U16";
    case VEC4_I32:
        return "VEC4_I32";
    case VEC4_U32:
        return "VEC4_U32";
    case VEC4_FLOAT:
        return "VEC4_FLOAT";
    case VEC4_DOUBLE:
        return "VEC4_DOUBLE";
    case MAT2_I8:
        return "MAT2_I8";
    case MAT2_U8:
        return "MAT2_U8";
    case MAT2_I16:
        return "MAT2_I16";
    case MAT2_U16:
        return "MAT2_U16";
    case MAT2_I32:
        return "MAT2_I32";
    case MAT2_U32:
        return "MAT2_U32";
    case MAT2_FLOAT:
        return "MAT2_FLOAT";
    case MAT2_DOUBLE:
        return "MAT2_DOUBLE";
    case MAT3_I8:
        return "MAT3_I8";
    case MAT3_U8:
        return "MAT3_U8";
    case MAT3_I16:
        return "MAT3_I16";
    case MAT3_U16:
        return "MAT3_U16";
    case MAT3_I32:
        return "MAT3_I32";
    case MAT3_U32:
        return "MAT3_U32";
    case MAT3_FLOAT:
        return "MAT3_FLOAT";
    case MAT3_DOUBLE:
        return "MAT3_DOUBLE";
    case MAT4_I8:
        return "MAT4_I8";
    case MAT4_U8:
        return "MAT4_U8";
    case MAT4_I16:
        return "MAT4_I16";
    case MAT4_U16:
        return "MAT4_U16";
    case MAT4_I32:
        return "MAT4_I32";
    case MAT4_U32:
        return "MAT4_U32";
    case MAT4_FLOAT:
        return "MAT4_FLOAT";
    case MAT4_DOUBLE:
        return "MAT4_DOUBLE";
    default:
        return "UNKNOWN DATATYPE";
    }
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
            return DTYPE::INT8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return DTYPE::UINT8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return DTYPE::INT16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return DTYPE::UINT16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return DTYPE::INT32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return DTYPE::UINT32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return DTYPE::FLOAT;
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
    case DTYPE::INT8:
    case DTYPE::UINT8:
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
    case DTYPE::INT16:
    case DTYPE::UINT16:
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
    case DTYPE::INT32:
    case DTYPE::UINT32:
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
    case DTYPE::FLOAT:
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
    }
}

size_t dtype_components(DTYPE type)
{
    switch (type) {
    case DTYPE::INT8:
    case DTYPE::UINT8:
    case DTYPE::INT16:
    case DTYPE::UINT16:
    case DTYPE::INT32:
    case DTYPE::UINT32:
    case DTYPE::FLOAT:
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
