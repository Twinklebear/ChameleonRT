#include <cmath>
#include <glm/ext.hpp>
#include <glm/gtx/transform.hpp>
#include "arcball_camera.h"

/*
 * Project the point in [-1, 1] screen space onto the arcball sphere
 */
static glm::quat screen_to_arcball(const glm::vec2 &p);

ArcballCamera::ArcballCamera(const glm::vec3 &center, float motion_speed, const std::array<int, 2> &screen)
	: motion_speed(motion_speed), inv_screen({1.f / screen[0], 1.f / screen[1]})
{
	center_translation = glm::inverse(glm::translate(center));
	translation = glm::translate(glm::vec3(0.f, 0.f, -1.f));
	rotation = glm::quat(1.f, 0.f, 0.f, 0.f);

	update_camera();
}
bool ArcballCamera::mouse(const SDL_Event &mouse, float elapsed){
	if (mouse.type == SDL_MOUSEMOTION) {
		auto motion = mouse.motion;
		if (motion.state & SDL_BUTTON_LMASK) {
			rotate(motion);
			return true;
		} else if (motion.state & SDL_BUTTON_RMASK) {
			pan(motion);
			return true;
		}
	} else if (mouse.type == SDL_MOUSEWHEEL) {
		auto scroll = mouse.wheel;
		if (scroll.y != 0){
			glm::vec3 motion{0.f};
			motion.z = scroll.y * 0.35;
			translation = glm::translate(motion * motion_speed * elapsed) * translation;
			update_camera();
			return true;
		}
	}
	return false;
}
const glm::mat4& ArcballCamera::transform() const {
	return camera;
}
const glm::mat4& ArcballCamera::inv_transform() const {
	return inv_camera;
}
glm::vec3 ArcballCamera::eye_pos() const {
	return glm::vec3{inv_camera * glm::vec4{0, 0, 0, 1}};
}
glm::vec3 ArcballCamera::eye_dir() const {
	return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 0, -1, 0}});
}
glm::vec3 ArcballCamera::up_dir() const {
	return glm::normalize(glm::vec3{inv_camera * glm::vec4{0, 1, 0, 0}});
}
void ArcballCamera::rotate(const SDL_MouseMotionEvent &mouse) {
	// Compute current and previous mouse positions in clip space
	glm::vec2 mouse_cur = glm::vec2{mouse.x * 2.0 * inv_screen[0] - 1.0,
		1.0 - 2.0 * mouse.y * inv_screen[1]};
	glm::vec2 mouse_prev = glm::vec2{(mouse.x - mouse.xrel) * 2.0 * inv_screen[0] - 1.0,
		1.0 - 2.0 * (mouse.y - mouse.yrel) * inv_screen[1]};
	// Clamp mouse positions to stay in screen space range
	mouse_cur = glm::clamp(mouse_cur, glm::vec2{-1, -1}, glm::vec2{1, 1});
	mouse_prev = glm::clamp(mouse_prev, glm::vec2{-1, -1}, glm::vec2{1, 1});
	glm::quat mouse_cur_ball = screen_to_arcball(mouse_cur);
	glm::quat mouse_prev_ball = screen_to_arcball(mouse_prev);

	rotation = mouse_cur_ball * mouse_prev_ball * rotation;
	update_camera();
}
void ArcballCamera::pan(const SDL_MouseMotionEvent &mouse){
	const float zoom_amount = std::abs(translation[3][2]);
	glm::vec4 motion(mouse.xrel * inv_screen[0] * zoom_amount,
			-mouse.yrel * inv_screen[1] * zoom_amount,
			0.f, 0.f);
	// Find the panning amount in the world space
	motion = inv_camera * motion;
	center_translation = glm::translate(glm::vec3(motion)) * center_translation;
	update_camera();
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

