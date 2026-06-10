//
// Created by Oleki on 05.06.2026.
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../include/Renderer.h"
#include <iostream>
#include <thread>

#include "Config.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Particle.h"
#include "ParticleGenerator.h"


void Renderer::init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);  // setup major version of GLFW same as OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);  // setup minor version of GLFW same as OpenGL
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // use core profile of GLFW to access a smaller subset of oGL without backwards compatibility

    window = glfwCreateWindow(800, 600, "Barnes-Hut", nullptr, nullptr);    // initialize window object
    glfwMakeContextCurrent(window); // switch context to the window
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);    // mouse ENABLE

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))    // load openGL function pointers
    {
        std::cout << "Failed to initialize GL Loader-Generator (GLAD)" << std::endl;
        return;
    }

    //glfwSwapInterval(0);    // 0 = VSync Off
    glViewport(0, 0, 800, 600); // set the size of the viewport area (lower-left corner, width, height)
    // glfwSetWindowUserPointer(window, this); // set pointer of specified window
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  // window resize callback
    glfwSetCursorPosCallback(window, mouse_callback);                   // mouse coursor callback
    glfwSetWindowUserPointer(window, this);                       // set pointer to the current window

    // VAO  (Vertex Array Object)
    glGenVertexArrays(1, &VAO); // creating 1 buffer and saving its ID to VAO
    glBindVertexArray(VAO); // setting this VAO as the active one

    glGenBuffers(1, &VBO);  // generate 1 Vertex Buffer Object
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // set VBO as an active buffer for further operations / function calls
    glBufferData(GL_ARRAY_BUFFER,
        particles->size() * sizeof(Particle),
        particles->data(),
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

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    glPointSize(4.0f);

    // IMGUI INIT
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    glViewport(0, 0, width, height);
}

void Renderer::mouse_callback(GLFWwindow* window, double x, double y)
{
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));

    if (!renderer->mouseCaptured) return;

    renderer->camera.mouseInput((float)x, (float)y);
}

void Renderer::processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        mouseCaptured = false;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if (ImGui::GetIO().WantCaptureMouse) return;
        if (!mouseCaptured) {
            camera.mouseMoved = true;
        }
        mouseCaptured = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void Renderer::initFrame() {
    if (glfwWindowShouldClose(window)) {
        isTerminated = true;
        return;
    }

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    processInput(window);               // keyboard input
    camera.update(window, deltaTime);   // process camera

    // Clear off window's context
    glClearColor(0,0,0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // new frame form IMGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Renderer::renderFrame() {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(VAO);
    shader.use();

    // if size of particles changed - reallocate buffer
    if (particles->size() != lastParticleCount) {
        glBufferData(GL_ARRAY_BUFFER, particles->size() * sizeof(Particle), particles->data(), GL_DYNAMIC_DRAW);
        lastParticleCount = particles->size();
    } else {    // if size is same
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles->size() * sizeof(Particle), particles->data());
    }

    glm::mat4 modelMatrix(1.0f);    // identity matrix
    glm::mat4 viewMatrix = camera.getViewMatrix();
    glm::mat4 projectionMatrix = camera.getProjectionMatrix(800.0f / 600.0f);

    unsigned int model = glGetUniformLocation(shader.ID, "model");
    unsigned int view = glGetUniformLocation(shader.ID, "view");
    unsigned int projection = glGetUniformLocation(shader.ID, "projection");

    // Sending to the shader's uniforms model, view and projection matrices
    glUniformMatrix4fv(model,1,GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(view,1,GL_FALSE,glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projection,1,GL_FALSE,glm::value_ptr(projectionMatrix));

    // send to GPU updated particles data
    glBufferSubData(GL_ARRAY_BUFFER,
                     0,
                     particles->size() * sizeof(Particle),
                     particles->data());

    // render sent particles
    glDrawArrays(GL_POINTS, 0, particles->size());

    // render IMGUI window
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::prepareImGuiFrame() {
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 10.0f, 10.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::Begin("Config", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

    // ##### CONFIG #####

    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Config variables:");
    //ImGui::SliderFloat("G Multiplier", &G_MULTIPLIER, 0.0f, 300.0f);
    ImGui::InputFloat("G Multiplier", &G_MULTIPLIER, 1.0f, 10.0f, "%.1f");

    if (ImGui::SliderFloat("Theta", &THETA, 0.0f, 5.0f)) THETA_SQ = THETA * THETA;
    if (ImGui::SliderFloat("Epsilon", &EPSILON, 0.01f, 1.0f)) EPSILON_SQ = EPSILON * EPSILON;


    //ImGui::SliderFloat("Timestep", &TIME_STEP, 0.0f, 50000.0f);
    ImGui::InputFloat("Timestep", &TIME_STEP, 1.0f, 100.0f, "%.1f");
    ImGui::SliderInt("Threads", &NUM_THREADS, 1, MAX_HARDWARE_THREADS);

    // ##### CONFIG #####

    ImGui::Separator();
    ImGui::Text("TPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Current Bodies: %zu", particles->size());

    if (ImGui::Button("Remove Bodies", ImVec2(-1, 0))) {
        particles->clear();
    }
    ImGui::Separator();

    // ##### PARTICLE GENERATOR #####

    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Particle Generator:");

    static float genParticleMass = 1.0f;
    static float genCenterMass = 150000.0f;
    static int genCount = 10000;

    ImGui::InputFloat("Particle Mass", &genParticleMass, 10.0f, 100.0f, "%.1f");
    ImGui::InputFloat("Center Mass", &genCenterMass, 1000.0f, 10000.0f, "%.1f");
    ImGui::InputInt("Count", &genCount, 100, 1000);

    glm::vec3 FOC = camera.position + camera.viewDirection * 150.0f;      // front of camera
    glm::vec3 CVV = camera.getVelocityVector() * (deltaTime / TIME_STEP); // camera velocity vector

    if (ImGui::Button("Create Particle", ImVec2(-1, 0))) {
        ParticleGenerator::addParticle(*particles, FOC.x, FOC.y, FOC.z, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Create Rectangle", ImVec2(-1, 0))) {
        ParticleGenerator::createFlatRectangle(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Create Cube", ImVec2(-1, 0))) {
        ParticleGenerator::createCube(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Create Disc", ImVec2(-1, 0))) {
        ParticleGenerator::createDisc(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, genCenterMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Create Sphere", ImVec2(-1, 0))) {
        ParticleGenerator::createSphere(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, genCenterMass, CVV.x, CVV.y, CVV.z);
    }

    // ##### PARTICLE GENERATOR #####


    // ##### CAMERA #####

    ImGui::Separator();

    if (ImGui::Button("Reset position", ImVec2(-1, 0))) {
        camera.position = glm::vec3(0,0,0);
    }
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Camera:");
    ImGui::Text("Speed: %.1f", camera.speed);
    ImGui::Text("Pitch: %.1f", camera.pitch);
    ImGui::Text("Yaw: %.1f", camera.yaw);
    ImGui::Text("Cam Pos: %.2f, %.2f, %.2f", camera.position.x, camera.position.y, camera.position.z);

    // ##### CAMERA #####

    ImGui::End();
}


glm::vec3 Renderer::getCameraXYZ() {
    return camera.position;
}




