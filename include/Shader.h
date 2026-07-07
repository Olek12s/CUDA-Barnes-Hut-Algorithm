//
// Created by Oleki on 05.06.2026.
//

#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <iostream>
#include <fstream>

#include "glm/gtc/type_ptr.hpp"

#define uint unsigned int

class Shader {
public:
    uint ID;


    Shader() {}

    Shader(const std::string& shaderVex, const std::string& shaderFrag) {
        std::ifstream vexFile("../shaders/" + shaderVex);
        std::ifstream fragFile("../shaders/" + shaderFrag);

        bool loadFail = false;
        if (!vexFile.is_open()) {
            std::cout << "Failed to read vertex shader file: " << shaderVex << "\n";
            loadFail = true;
        }

        if (!fragFile.is_open()) {
            std::cout << "Failed to read fragment shader file: " << shaderFrag << "\n";
            loadFail = true;
        }
        if (loadFail) return;

        std::string vexCode;
        std::string fragCode;
        std::string line;

        while (std::getline(vexFile, line)) {
            vexCode.append(line);
            vexCode.append("\n");
        }

        while (std::getline(fragFile, line)) {
            fragCode.append(line);
            fragCode.append("\n");
        }

        vexFile.close();
        fragFile.close();

        std::cout << "Vertex shader loaded (" << vexCode.size() << " bytes)\n";
        std::cout << "Fragment shader loaded (" << fragCode.size() << " bytes)\n";


        // ##### shaders compilation ##### //

        // ##### Vertex Shader ##### //
        uint vex;
        int ok;
        char logs[1024];
        const char* srcVex = vexCode.c_str();
        const char* srcFrag = fragCode.c_str();

        vex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vex, 1, &srcVex, nullptr);
        glCompileShader(vex);
        glGetShaderiv(vex, GL_COMPILE_STATUS, &ok);

        if (!ok) {
            glGetShaderInfoLog(vex, 1024, nullptr, logs);
            std::cout << "Failed to compile Vertex Shaders: " << logs << "\n";
        }

        // ##### Fragment Shader ##### //

        uint frag;
        frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &srcFrag, nullptr);
        glCompileShader(frag);
        glGetShaderiv(frag, GL_COMPILE_STATUS, &ok);

        if (!ok) {
            glGetShaderInfoLog(frag, 1024, nullptr, logs);
            std::cout << "Failed to compiled Fragment Shaders: " << logs << "\n";
            return;
        }

        // ##### Creating shader program ##### //

        ID = glCreateProgram();
        glAttachShader(ID, vex);
        glAttachShader(ID, frag);
        glLinkProgram(ID);

        glGetProgramiv(ID, GL_LINK_STATUS, &ok);
        if (!ok) {
            glGetProgramInfoLog(ID, 1024, nullptr, logs);
            std::cout << "Failed to link shaders: " << logs << "\n";
            return;
        }

        glDeleteShader(vex);
        glDeleteShader(frag);

        std::cout << "\n########################################\n";
        std::cout << "Shaders building ended successfully.\n";
        std::cout << "########################################\n\n";
    }

    void use() {
        glUseProgram(ID);
    }

    void setBool(const std::string& name, bool value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }

    void setInt(const std::string& name, int value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setFloat(const std::string& name, float value) const {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setMat4(const std::string& name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
    }
};

#endif //SHADER_H
