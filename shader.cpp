#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <regex>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "shader.h"

// Load a GLSL shader from the file. Returns -1 if loading fails and prints
// out the compilation errors
GLint compile_shader(GLenum type, const std::string &src);

Shader::Shader(const std::string &vert_src, const std::string &frag_src) {
	GLint vert = compile_shader(GL_VERTEX_SHADER, vert_src);
	if (vert == -1) {
		throw std::runtime_error("Failed to compile vertex shader");
	}

	GLint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src);
	if (frag == -1) {
		throw std::runtime_error("Failed to compile fragment shader");
	}

	program = glCreateProgram();
	glAttachShader(program, vert);
	glAttachShader(program, frag);
	glLinkProgram(program);
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		std::cout << "Error loading shader program: Program failed to link, log:\n";
		GLint len;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
		std::vector<char> log(len, '\0');
		log.resize(len);
		glGetProgramInfoLog(program, log.size(), 0, log.data());
		std::cout << log.data() << "\n";
	}

	glDetachShader(program, vert);
	glDetachShader(program, frag);
	glDeleteShader(vert);
	glDeleteShader(frag);

	if (status == GL_FALSE){
		glDeleteProgram(program);
		throw std::runtime_error("Failed to link program");
	}

	parse_uniforms(vert_src);
	parse_uniforms(frag_src);
}

Shader::~Shader() {
	glDeleteProgram(program);
}

template<>
void Shader::uniform<bool>(const std::string &unif, const bool &t) {
	glUniform1i(uniforms[unif], t ? 1 : 0);
}

template<>
void Shader::uniform<int>(const std::string &unif, const int &t) {
	glUniform1i(uniforms[unif], t);
}

template<>
void Shader::uniform<float>(const std::string &unif, const float &t) {
	glUniform1f(uniforms[unif], t);
}

template<>
void Shader::uniform<glm::vec3>(const std::string &unif, const glm::vec3 &t) {
	glUniform3fv(uniforms[unif], 1, &t.x);
}

template<>
void Shader::uniform<glm::mat4>(const std::string &unif, const glm::mat4 &t) {
	glUniformMatrix4fv(uniforms[unif], 1, GL_FALSE, glm::value_ptr(t));
}

void Shader::parse_uniforms(const std::string &src) {
	const std::regex regex_unif("uniform[^;]+[ ](\\w+);");
	for (auto it = std::sregex_iterator(src.begin(), src.end(), regex_unif);
			it != std::sregex_iterator(); ++it)
	{
		const std::smatch &m = *it;
		uniforms[m[1]] = glGetUniformLocation(program, m[1].str().c_str());
	}
}

GLint compile_shader(GLenum type, const std::string &src) {
	GLuint shader = glCreateShader(type);
	const char *csrc = src.c_str();
	glShaderSource(shader, 1, &csrc, 0);
	glCompileShader(shader);
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		std::cout << "Shader compilation error, ";
		switch (type){
			case GL_VERTEX_SHADER:
				std::cout << "Vertex shader: ";
				break;
			case GL_FRAGMENT_SHADER:
				std::cout << "Fragment shader: ";
				break;
			case GL_GEOMETRY_SHADER:
				std::cout << "Geometry shader: ";
				break;
			case GL_COMPUTE_SHADER:
				std::cout << "Compute shader: ";
				break;
			case GL_TESS_CONTROL_SHADER:
				std::cout << "Tessellation Control shader: ";
				break;
			case GL_TESS_EVALUATION_SHADER:
				std::cout << "Tessellation Evaluation shader: ";
				break;
			default:
				std::cout << "Unknown shader type: ";
		}
		GLint len;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
		std::vector<char> log(len, '\0');
		log.resize(len);
		glGetShaderInfoLog(shader, log.size(), 0, log.data());
		glDeleteShader(shader);
		return -1;
	}
	return shader;
}

