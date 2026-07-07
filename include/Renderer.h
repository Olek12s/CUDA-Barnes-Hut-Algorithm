//
// Created by Oleki on 05.06.2026.
//

#ifndef RENDERER_H
#define RENDERER_H

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Camera.h"
#include "Octree.h"
#include "Particle.h"
#include "Shader.h"

class Renderer {
    std::vector<Particle>* particles;
    Octree* octree;
    GLFWwindow* window;
    Shader shader;
    unsigned int VBO;   // Vertex Buffer Object
    unsigned int VAO;   // Vertex Array Object

    bool mouseCaptured = true;
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    int lastParticleCount = 0;

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void mouse_callback(GLFWwindow* window, double x, double y);
    void processInput(GLFWwindow* window);
public:
    Renderer(std::vector<Particle> &particles, Octree &octtree): particles(&particles), octree(&octtree)  {}
    void init();
    void initFrame();
    void prepareImGuiFrame();
    void renderFrame();

    Camera camera;
    bool isTerminated = false;
    float currentTPS = 0.0f;
};



#endif //RENDERER_H
