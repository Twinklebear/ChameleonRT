#pragma once

#include <unordered_map>
#include <string>
#include "gl_core_4_5.h"

struct Shader {
	GLuint program;
	std::unordered_map<std::string, GLint> uniforms;

	Shader(const std::string &vert_src, const std::string &frag_src);
	~Shader();
	template<typename T>
	void uniform(const std::string &unif, const T &t);

private:
	// Parse the uniform variable declarations in the src file and
	// add them to the uniforms map
	void parse_uniforms(const std::string &src);
};

