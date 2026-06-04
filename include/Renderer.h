//
// Created by Oleki on 05.06.2026.
//

#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Renderer {
    GLFWwindow* window;

public:
    void initGLFW();
    void framebuffer_size_callback(GLFWwindow* window, int width, int height);  // callback each time a window is resized
};



#endif //RENDERER_H
