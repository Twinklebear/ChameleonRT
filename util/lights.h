#pragma once

#include <glm/glm.hpp>

struct QuadLight {
	glm::vec4 emission;
	glm::vec4 position;
	glm::vec4 normal;

	glm::vec3 v_x;
	float width;

	glm::vec3 v_y;
	float height;
};

