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
    case DTYPE::INT_8:
        return "INT_8";
    case DTYPE::UINT_8:
        return "UINT_8";
    case DTYPE::INT_16:
        return "INT_16";
    case DTYPE::UINT_16:
        return "UINT_16";
    case DTYPE::INT_32:
        return "INT_32";
    case DTYPE::UINT_32:
        return "UINT_32";
    case DTYPE::FLOAT_32:
        return "FLOAT_32";
    case FLOAT_64:
        return "FLOAT_64";
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
    case VEC2_F32:
        return "VEC2_F32";
    case VEC2_F64:
        return "VEC2_F64";
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
    case VEC3_F32:
        return "VEC3_F32";
    case VEC3_F64:
        return "VEC3_F64";
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
    case VEC4_F32:
        return "VEC4_F32";
    case VEC4_F64:
        return "VEC4_F64";
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
    case MAT2_F32:
        return "MAT2_F32";
    case MAT2_F64:
        return "MAT2_F64";
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
    case MAT3_F32:
        return "MAT3_F32";
    case MAT3_F64:
        return "MAT3_F64";
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
    case MAT4_F32:
        return "MAT4_F32";
    case MAT4_F64:
        return "MAT4_F64";
    default:
        return "UNKNOWN DATATYPE";
    }
}

DTYPE parse_dtype(const std::string &str)
{
    if (str == "INT_8") {
        return DTYPE::INT_8;
    } else if (str == "UINT_8") {
        return DTYPE::UINT_8;
    } else if (str == "INT_16") {
        return DTYPE::INT_16;
    } else if (str == "UINT_16") {
        return DTYPE::UINT_16;
    } else if (str == "INT_32") {
        return DTYPE::INT_32;
    } else if (str == "UINT_32") {
        return DTYPE::UINT_32;
    } else if (str == "FLOAT_32") {
        return DTYPE::FLOAT_32;
    } else if (str == "FLOAT_64") {
        return DTYPE::FLOAT_64;
    } else if (str == "VEC2_I8") {
        return DTYPE::VEC2_I8;
    } else if (str == "VEC2_U8") {
        return DTYPE::VEC2_U8;
    } else if (str == "VEC2_I16") {
        return DTYPE::VEC2_I16;
    } else if (str == "VEC2_U16") {
        return DTYPE::VEC2_U16;
    } else if (str == "VEC2_I32") {
        return DTYPE::VEC2_I32;
    } else if (str == "VEC2_U32") {
        return DTYPE::VEC2_U32;
    } else if (str == "VEC2_F32") {
        return DTYPE::VEC2_F32;
    } else if (str == "VEC2_F64") {
        return DTYPE::VEC2_F64;
    } else if (str == "VEC3_I8") {
        return DTYPE::VEC3_I8;
    } else if (str == "VEC3_U8") {
        return DTYPE::VEC3_U8;
    } else if (str == "VEC3_I16") {
        return DTYPE::VEC3_I16;
    } else if (str == "VEC3_U16") {
        return DTYPE::VEC3_U16;
    } else if (str == "VEC3_I32") {
        return DTYPE::VEC3_I32;
    } else if (str == "VEC3_U32") {
        return DTYPE::VEC3_U32;
    } else if (str == "VEC3_F32") {
        return DTYPE::VEC3_F32;
    } else if (str == "VEC3_F64") {
        return DTYPE::VEC3_F64;
    } else if (str == "VEC4_I8") {
        return DTYPE::VEC4_I8;
    } else if (str == "VEC4_U8") {
        return DTYPE::VEC4_U8;
    } else if (str == "VEC4_I16") {
        return DTYPE::VEC4_I16;
    } else if (str == "VEC4_U16") {
        return DTYPE::VEC4_U16;
    } else if (str == "VEC4_I32") {
        return DTYPE::VEC4_I32;
    } else if (str == "VEC4_U32") {
        return DTYPE::VEC4_U32;
    } else if (str == "VEC4_F32") {
        return DTYPE::VEC4_F32;
    } else if (str == "VEC4_F64") {
        return DTYPE::VEC4_F64;
    } else if (str == "MAT2_I8") {
        return DTYPE::MAT2_I8;
    } else if (str == "MAT2_U8") {
        return DTYPE::MAT2_U8;
    } else if (str == "MAT2_I16") {
        return DTYPE::MAT2_I16;
    } else if (str == "MAT2_U16") {
        return DTYPE::MAT2_U16;
    } else if (str == "MAT2_I32") {
        return DTYPE::MAT2_I32;
    } else if (str == "MAT2_U32") {
        return DTYPE::MAT2_U32;
    } else if (str == "MAT2_F32") {
        return DTYPE::MAT2_F32;
    } else if (str == "MAT2_F64") {
        return DTYPE::MAT2_F64;
    } else if (str == "MAT3_I8") {
        return DTYPE::MAT3_I8;
    } else if (str == "MAT3_U8") {
        return DTYPE::MAT3_U8;
    } else if (str == "MAT3_I16") {
        return DTYPE::MAT3_I16;
    } else if (str == "MAT3_U16") {
        return DTYPE::MAT3_U16;
    } else if (str == "MAT3_I32") {
        return DTYPE::MAT3_I32;
    } else if (str == "MAT3_U32") {
        return DTYPE::MAT3_U32;
    } else if (str == "MAT3_F32") {
        return DTYPE::MAT3_F32;
    } else if (str == "MAT3_F64") {
        return DTYPE::MAT3_F64;
    } else if (str == "MAT4_I8") {
        return DTYPE::MAT4_I8;
    } else if (str == "MAT4_U8") {
        return DTYPE::MAT4_U8;
    } else if (str == "MAT4_I16") {
        return DTYPE::MAT4_I16;
    } else if (str == "MAT4_U16") {
        return DTYPE::MAT4_U16;
    } else if (str == "MAT4_I32") {
        return DTYPE::MAT4_I32;
    } else if (str == "MAT4_U32") {
        return DTYPE::MAT4_U32;
    } else if (str == "MAT4_F32") {
        return DTYPE::MAT4_F32;
    } else if (str == "MAT4_F64") {
        return DTYPE::MAT4_F64;
    } else {
        throw std::runtime_error("Invalid data type string: " + str);
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
            return DTYPE::INT_8;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return DTYPE::UINT_8;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return DTYPE::INT_16;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return DTYPE::UINT_16;
        case TINYGLTF_COMPONENT_TYPE_INT:
            return DTYPE::INT_32;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return DTYPE::UINT_32;
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return DTYPE::FLOAT_32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return FLOAT_64;
        default:
            break;
        };
        break;
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
            return VEC2_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC2_F64;
        default:
            break;
        };
        break;
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
            return VEC3_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC3_F64;
        default:
            break;
        };
        break;
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
            return VEC4_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return VEC4_F64;
        default:
            break;
        };
        break;
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
            return MAT2_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT2_F64;
        default:
            break;
        };
        break;
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
            return MAT3_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT3_F64;
        default:
            break;
        };
        break;
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
            return MAT4_F32;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return MAT4_F64;
        default:
            break;
        };
        break;
    default:
        break;
    }
    throw std::runtime_error("Unrecognized type/component type pair");
}

size_t dtype_stride(DTYPE type)
{
    switch (type) {
    case DTYPE::INT_8:
    case DTYPE::UINT_8:
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
    case DTYPE::INT_16:
    case DTYPE::UINT_16:
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
    case DTYPE::INT_32:
    case DTYPE::UINT_32:
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
    case DTYPE::FLOAT_32:
    case VEC2_F32:
    case VEC3_F32:
    case VEC4_F32:
    case MAT2_F32:
    case MAT3_F32:
    case MAT4_F32:
        return dtype_components(type) * 4;
    case FLOAT_64:
    case VEC2_F64:
    case VEC3_F64:
    case VEC4_F64:
    case MAT2_F64:
    case MAT3_F64:
    case MAT4_F64:
        return dtype_components(type) * 8;
    default:
        break;
    }
    throw std::runtime_error("UNKOWN DATATYPE");
}

size_t dtype_components(DTYPE type)
{
    switch (type) {
    case DTYPE::INT_8:
    case DTYPE::UINT_8:
    case DTYPE::INT_16:
    case DTYPE::UINT_16:
    case DTYPE::INT_32:
    case DTYPE::UINT_32:
    case DTYPE::FLOAT_32:
    case FLOAT_64:
        return 1;
    case VEC2_I8:
    case VEC2_U8:
    case VEC2_I16:
    case VEC2_U16:
    case VEC2_I32:
    case VEC2_U32:
    case VEC2_F32:
    case VEC2_F64:
        return 2;
    case VEC3_I8:
    case VEC3_U8:
    case VEC3_I16:
    case VEC3_U16:
    case VEC3_I32:
    case VEC3_U32:
    case VEC3_F32:
    case VEC3_F64:
        return 3;
    case VEC4_I8:
    case VEC4_U8:
    case VEC4_I16:
    case VEC4_U16:
    case VEC4_I32:
    case VEC4_U32:
    case VEC4_F32:
    case VEC4_F64:
    case MAT2_I8:
    case MAT2_U8:
    case MAT2_I16:
    case MAT2_U16:
    case MAT2_I32:
    case MAT2_U32:
    case MAT2_F32:
    case MAT2_F64:
        return 4;
    case MAT3_I8:
    case MAT3_U8:
    case MAT3_I16:
    case MAT3_U16:
    case MAT3_I32:
    case MAT3_U32:
    case MAT3_F32:
    case MAT3_F64:
        return 9;
    case MAT4_I8:
    case MAT4_U8:
    case MAT4_I16:
    case MAT4_U16:
    case MAT4_I32:
    case MAT4_U32:
    case MAT4_F32:
    case MAT4_F64:
        return 16;
    default:
        break;
    }
    throw std::runtime_error("Invalid data type");
}
