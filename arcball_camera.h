#pragma once

#include <array>
#include <SDL.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

/*
 * A simple arcball camera that moves around the camera's focal point
 * Controls:
 * left mouse + drag: rotate camera round focal point
 * right mouse + drag: translate camera
 * shift + right mouse + drag +/- y: zoom in/out
 * r key: reset camera to original look_at matrix
 */
class ArcballCamera {
	float motion_speed;
	// Inverse x, y window dimensions
	std::array<float, 2> inv_screen;
	// We store the unmodified look at matrix along with
	// decomposed translation and rotation components
	glm::mat4 center_translation, translation;
	glm::quat rotation;
	// camera is the full camera transform,
	// inv_camera is stored as well to easily compute
	// eye position and world space rotation axes
	glm::mat4 camera, inv_camera;

public:
	/*
	 * Create an arcball camera focused on some center point
	 * zoom speed: units per second speed of panning the camera
	 * screen: { WIN_X_SIZE, WIN_Y_SIZE }
	 */
	ArcballCamera(const glm::vec3 &center, float zoom_speed,
			const std::array<int, 2> &screen);
	/*
	 * Handle mouse events to move the camera
	 * returns true if the camera has changed
	 */
	bool mouse(const SDL_Event &mouse, float elapsed);
	/*
	 * Get the camera transformation matrix
	 */
	const glm::mat4& transform() const;
	/*
	 * Get the camera inverse transformation matrix
	 */
	const glm::mat4& inv_transform() const;
	/*
	 * Get the eye position of the camera in world space
	 */
	glm::vec3 eye_pos() const;
	// Get the eye direction of the camera in world space
	glm::vec3 eye_dir() const;
	// Get the up direction of the camera in world space
	glm::vec3 up_dir() const;

private:
	/*
	 * Handle rotation events
	 */
	void rotate(const SDL_MouseMotionEvent &mouse);
	/*
	 * Handle panning/zooming events
	 */
	void pan(const SDL_MouseMotionEvent &mouse);

	void update_camera();
};

