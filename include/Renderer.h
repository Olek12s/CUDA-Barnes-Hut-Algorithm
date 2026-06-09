//
// Created by Oleki on 05.06.2026.
//

#ifndef RENDERER_H
#define RENDERER_H

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Camera.h"
#include "Particle.h"
#include "Shader.h"

class Renderer {
private:
    std::vector<Particle>& particles;
    GLFWwindow* window;
    Shader shader;
    unsigned int VBO;   // Vertex Buffer Object
    unsigned int VAO;   // Vertex Array Object
    Camera camera;
    bool mouseCaptured = true;

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);  // callback each time a window is resized
    static void mouse_callback(GLFWwindow* window, double x, double y);
    void processInput(GLFWwindow* window);
public:
    Renderer(std::vector<Particle> &particles): particles(particles) {}
    void init();
    void frameTick();
    glm::vec3 getCameraXYZ();

    bool isTerminated = false;
};



#endif //RENDERER_H
