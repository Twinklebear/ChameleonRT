#include <array>
#include <algorithm>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <glm/ext.hpp>
#include "util.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::string pretty_print_count(const double count) {
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

uint64_t align_to(uint64_t val, uint64_t align) {
	return ((val + align - 1) / align) * align;
}

void ortho_basis(glm::vec3 &v_x, glm::vec3 &v_y, const glm::vec3 &n) {
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

void canonicalize_path(std::string &path) {
	std::replace(path.begin(), path.end(), '\\', '/');
}

std::string get_cpu_brand() {
	std::string brand = "Unspecified";
	std::array<int, 4> regs;
	__cpuid(regs.data(), 0x80000000);
	if (regs[0] >= 0x80000004) {
		char b[64] = {0};
		for (int i = 0; i < 3; ++i) {
			__cpuid(regs.data(), 0x80000000 + i + 2);
			std::memcpy(b + i * sizeof(regs), regs.data(), sizeof(regs));
		}
		brand = b;
	}
	return brand;
}

