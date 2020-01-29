#include <iostream>
#include <algorithm>
#include <array>
#include <signal.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include "util.h"
#include <glm/ext.hpp>
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef ENABLE_OPEN_IMAGE_DENOISE
#include <OpenImageDenoise/oidn.hpp>

void oidn_error_callback(void* userPtr, oidn::Error error, const char* message)
{
    throw std::runtime_error(message);
}

volatile bool oidn_is_cancelled = false;
static oidn::DeviceRef device;

void oidn_signal_handler(int signal)
{
  oidn_is_cancelled = true;
}

bool oidn_progress_callback(void* userPtr, double n)
{
  if (oidn_is_cancelled)
    return false;
  std::cout << "\rDenoising " << int(n * 100.) << "%" << std::flush;
  return true;
}

void oidn_init()
{
    bool hdr = false;
    bool srgb = false;
    int numThreads = -1;
    int setAffinity = -1;
    int verbose = -1;
    
    device = oidn::newDevice();

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
      throw std::runtime_error(errorMessage);

    device.setErrorFunction(oidn_error_callback);

    if (numThreads > 0)
      device.set("numThreads", numThreads);
    if (setAffinity >= 0)
      device.set("setAffinity", bool(setAffinity));
    if (verbose >= 0)
      device.set("verbose", verbose);
    device.commit();    
}

// Note, input and output must be float3
void oidn_denoise(std::vector<float> &input, uint32_t width, uint32_t height, std::vector<float> &output) {
    int maxMemoryMB = -1;
    std::string filterType = "RT";
    oidn::FilterRef filter = device.newFilter(filterType.c_str());
    auto format = oidn::Format::Float3;
    
    filter.setImage("color", input.data(), format, width, height);
    // if (albedo)
    //     filter.setImage("albedo", albedo.getData(), format, width, height);
    // if (normal)
    //     filter.setImage("normal", normal.getData(), format, width, height);
    filter.setImage("output", output.data(), format, width, height);

    if (maxMemoryMB >= 0)
        filter.set("maxMemoryMB", maxMemoryMB);

    // Might want to change SIGINT behavior...
    filter.setProgressMonitorFunction(oidn_progress_callback);
    signal(SIGINT, oidn_signal_handler);

    filter.commit();

    // Start denoising
    filter.execute();

    filter.setProgressMonitorFunction(nullptr);
    signal(SIGINT, SIG_DFL);
}

void oidn_denoise(std::vector<uint32_t> &input, uint32_t width, uint32_t height, std::vector<uint32_t> &output) {
    assert(input.size() == (width * height));
    std::vector<float> float_in(input.size() * 3);
    std::vector<float> float_out(input.size() * 3);

    for (uint32_t i = 0; i < (width * height); ++i) {
        for (uint32_t c = 0; c < 3; ++c) {
            float_in[i * 3 + c] = ((uint8_t*) output.data())[i * 4 + c] / 255.f;
        }
    }
    
    oidn_denoise(float_in, width, height, float_out);

    for (uint32_t i = 0; i < (width * height); ++i) {
        for (uint32_t c = 0; c < 3; ++c) {
            ((uint8_t*) output.data())[i * 4 + c] = uint8_t(float_out[i * 3 + c] * 255.f);
        }
    }
}

#endif

std::string pretty_print_count(const double count)
{
    const double giga = 1000000000;
    const double mega = 1000000;
    const double kilo = 1000;
    if (count > giga) {
        return std::to_string(count / giga) + " G";
    } else if (count > mega) {
        return std::to_string(count / mega) + " M";
    } else if (count > kilo) {
        return std::to_string(count / kilo) + " K";
    }
    return std::to_string(count);
}

uint64_t align_to(uint64_t val, uint64_t align)
{
    return ((val + align - 1) / align) * align;
}

void ortho_basis(glm::vec3 &v_x, glm::vec3 &v_y, const glm::vec3 &n)
{
    v_y = glm::vec3(0);

    if (n.x < 0.6f && n.x > -0.6f) {
        v_y.x = 1.f;
    } else if (n.y < 0.6f && n.y > -0.6f) {
        v_y.y = 1.f;
    } else if (n.z < 0.6f && n.z > -0.6f) {
        v_y.z = 1.f;
    } else {
        v_y.x = 1.f;
    }
    v_x = glm::normalize(glm::cross(v_y, n));
    v_y = glm::normalize(glm::cross(n, v_x));
}

void canonicalize_path(std::string &path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
}

std::string get_file_extension(const std::string &fname)
{
    const size_t fnd = fname.find_last_of('.');
    if (fnd == std::string::npos) {
        return "";
    }
    return fname.substr(fnd + 1);
}

std::string get_cpu_brand()
{
    std::string brand = "Unspecified";
    std::array<int32_t, 4> regs;
#ifdef _WIN32
    __cpuid(regs.data(), 0x80000000);
#else
    __cpuid(0x80000000, regs[0], regs[1], regs[2], regs[3]);
#endif
    if (regs[0] >= 0x80000004) {
        char b[64] = {0};
        for (int i = 0; i < 3; ++i) {
#ifdef _WIN32
            __cpuid(regs.data(), 0x80000000 + i + 2);
#else
            __cpuid(0x80000000 + i + 2, regs[0], regs[1], regs[2], regs[3]);
#endif
            std::memcpy(b + i * sizeof(regs), regs.data(), sizeof(regs));
        }
        brand = b;
    }
    return brand;
}

float srgb_to_linear(float x)
{
    if (x <= 0.04045f) {
        return x / 12.92f;
    }
    return std::pow((x + 0.055f) / 1.055f, 2.4);
}

float linear_to_srgb(float x)
{
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}
