//
// Created by Oleki on 05.06.2026.
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../include/Renderer.h"
#include <iostream>


void Renderer::initGLFW() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);  // setup major version of GLFW same as OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);  // setup minor version of GLFW same as OpenGL
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // use core profile of GLFW to access a smaller subset of oGL without backwards compatibility

    window = glfwCreateWindow(800, 600, "Barnes-Hut", nullptr, nullptr);    // initialize window object
    glfwMakeContextCurrent(window); // switch context to the window

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))    // load openGL function pointers
    {
        std::cout << "Failed to initialize GL Loader-Generator (GLAD)" << std::endl;
        return;
    }
    glViewport(0, 0, 800, 600); // set the size of the viewport area (lower-left corner, width, height)
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


