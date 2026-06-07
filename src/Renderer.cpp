//
// Created by Oleki on 05.06.2026.
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../include/Renderer.h"
#include <iostream>

#include "Particle.h"


void Renderer::init() {
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
    // glfwSetWindowUserPointer(window, this); // set pointer of specified window
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // VAO  (Vertex Array Object)
    glGenVertexArrays(1, &VAO); // creating 1 buffer and saving its ID to VAO
    glBindVertexArray(VAO); // setting this VAO as the active one

    glGenBuffers(1, &VBO);  // generate 1 Vertex Buffer Object
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // set VBO as an active buffer for further operations / function calls
    glBufferData(GL_ARRAY_BUFFER,
        particles.size() * sizeof(Particle),
        particles.data(),
        GL_DYNAMIC_DRAW);  // allocating memory on GPU for Particle's vertices

    // telling openGL how to read data from VBO (position)
    glVertexAttribPointer(
        0,                              // index
        3,                              // size
        GL_FLOAT,                       // type
        GL_FALSE,                       // normalized
        sizeof(Particle),               // stride
        (void*)offsetof(Particle, x)    // offset
    );
    glEnableVertexAttribArray(0);

    // telling openGL how to read data from VBO (velocity)
    glVertexAttribPointer(
    1,                                  // index
    3,                                  // size
    GL_FLOAT,                           // type
    GL_FALSE,                           // normalized
    sizeof(Particle),                   // stride
    (void*)offsetof(Particle, vx)       // offset
    );
    glEnableVertexAttribArray(1);

    shader = Shader("Particle.vex", "Particle.frag");  // loading shader's files and compiling both vex and frag shaders
    shader.use();
}

void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    glViewport(0, 0, width, height);
}

void Renderer::processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
        isTerminated = true;
    }
}


void Renderer::frameTick() {
    if (glfwWindowShouldClose(window)) {
        isTerminated = true;
        return;
    }

    processInput(window);   // keyboard input
    update(window);         // process camera

    // ##### rendering calls ##### //
    glClearColor(0,0,0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);


    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(VAO);
    shader.use();

    // send to GPU updated particles data
    glBufferSubData(GL_ARRAY_BUFFER,
                     0,
                     particles.size() * sizeof(Particle),
                     particles.data());

    glPointSize(4.0f);
    glDrawArrays(GL_POINTS, 0, particles.size());


    glfwSwapBuffers(window);
    glfwPollEvents();

    std::cout << "#";
}





