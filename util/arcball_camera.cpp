#include <iostream>
#include <cmath>
#include <glm/ext.hpp>
#include <glm/gtx/transform.hpp>
#include "arcball_camera.h"

// Project the point in [-1, 1] screen space onto the arcball sphere
static glm::quat screen_to_arcball(const glm::vec2 &p);

ArcballCamera::ArcballCamera(const glm::vec3 &eye, const glm::vec3 &center,
		const glm::vec3 &up)
{
	const glm::vec3 dir = center - eye;
	glm::vec3 z_axis = glm::normalize(dir);
	glm::vec3 x_axis = glm::normalize(glm::cross(z_axis, glm::normalize(up)));
	glm::vec3 y_axis = glm::normalize(glm::cross(x_axis, z_axis));
	x_axis = glm::normalize(glm::cross(z_axis, y_axis));

	center_translation = glm::inverse(glm::translate(center));
	translation = glm::translate(glm::vec3(0.f, 0.f, -glm::length(dir)));
	rotation = glm::normalize(glm::quat_cast(glm::transpose(glm::mat3(x_axis, y_axis, -z_axis))));

	update_camera();
}
void ArcballCamera::rotate(glm::vec2 prev_mouse, glm::vec2 cur_mouse) {
	// Clamp mouse positions to stay in NDC
	cur_mouse = glm::clamp(cur_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});
	prev_mouse = glm::clamp(prev_mouse, glm::vec2{-1, -1}, glm::vec2{1, 1});

	const glm::quat mouse_cur_ball = screen_to_arcball(cur_mouse);
	const glm::quat mouse_prev_ball = screen_to_arcball(prev_mouse);

	rotation = mouse_cur_ball * mouse_prev_ball * rotation;
	update_camera();
}
void ArcballCamera::pan(glm::vec2 mouse_delta) {
	const float zoom_amount = std::abs(translation[3][2]);
	glm::vec4 motion(mouse_delta.x * zoom_amount, mouse_delta.y * zoom_amount, 0.f, 0.f);
	// Find the panning amount in the world space
	motion = inv_camera * motion;

	center_translation = glm::translate(glm::vec3(motion)) * center_translation;
	update_camera();
}
void ArcballCamera::zoom(const float zoom_amount) {
	const glm::vec3 motion(0.f, 0.f, zoom_amount);

	translation = glm::translate(motion) * translation;
	update_camera();
}
const glm::mat4& ArcballCamera::transform() const {
	return camera;
}
const glm::mat4& ArcballCamera::inv_transform() const {
	return inv_camera;
}
glm::vec3 ArcballCamera::eye() const {
	return glm::vec3{inv_camera * glm::vec4{0, 0, 0, 1}};
}
glm::vec3 ArcballCamera::dir() const {
	return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 0, -1, 0}});
}
glm::vec3 ArcballCamera::up() const {
	return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 1, 0, 0}});
}
glm::vec3 ArcballCamera::center() const {
	return -glm::column(center_translation, 3);
}
void ArcballCamera::update_camera() {
	camera = translation * glm::mat4_cast(rotation) * center_translation;
	inv_camera = glm::inverse(camera);
}

glm::quat screen_to_arcball(const glm::vec2 &p){
	const float dist = glm::dot(p, p);
	// If we're on/in the sphere return the point on it
	if (dist <= 1.f){
		return glm::quat(0.0, p.x, p.y, std::sqrt(1.f - dist));
	} else {
		// otherwise we project the point onto the sphere
		const glm::vec2 proj = glm::normalize(p);
		return glm::quat(0.0, proj.x, proj.y, 0.f);
	}
}

