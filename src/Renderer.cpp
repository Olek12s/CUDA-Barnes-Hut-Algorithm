//
// Created by Oleki on 05.06.2026.
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "../include/Renderer.h"
#include <iostream>
#include <thread>

#include "Globals.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Particle.h"
#include "ParticleGenerator.h"


void Renderer::init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "Barnes-Hut", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GL Loader-Generator (GLAD)" << std::endl;
        return;
    }

    //glfwSwapInterval(0);    // 0 = VSync Off
    glViewport(0, 0, 800, 600);
    // glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetWindowUserPointer(window, this);

    // VAO  (Vertex Array Object)
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
        particles->size() * sizeof(Particle),
        particles->data(),
        GL_DYNAMIC_DRAW);


    glVertexAttribPointer(
        0,                              // index
        3,                              // size
        GL_FLOAT,                       // type
        GL_FALSE,                       // normalized
        sizeof(Particle),               // stride
        (void*)offsetof(Particle, x)    // offset
    );
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
    1,                                  // index
    3,                                  // size
    GL_FLOAT,                           // type
    GL_FALSE,                           // normalized
    sizeof(Particle),                   // stride
    (void*)offsetof(Particle, vx)       // offset
    );
    glEnableVertexAttribArray(1);

    shader = Shader("Particle.vex", "Particle.frag");
    shader.use();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glPointSize(5.0f);

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

    processInput(window);
    camera.update(window, deltaTime);

    glClearColor(0,0,0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Renderer::renderFrame() {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(VAO);
    shader.use();

    if (particles->size() != lastParticleCount) {
        glBufferData(GL_ARRAY_BUFFER, particles->size() * sizeof(Particle), particles->data(), GL_DYNAMIC_DRAW);
        lastParticleCount = particles->size();
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles->size() * sizeof(Particle), particles->data());
    }

    glm::mat4 modelMatrix(1.0f);
    glm::mat4 viewMatrix = camera.getViewMatrix();
    glm::mat4 projectionMatrix = camera.getProjectionMatrix(800.0f / 600.0f);

    unsigned int model = glGetUniformLocation(shader.ID, "model");
    int view = glGetUniformLocation(shader.ID, "view");
    int projection = glGetUniformLocation(shader.ID, "projection");

    glUniformMatrix4fv(model,1,GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(view,1,GL_FALSE,glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projection,1,GL_FALSE,glm::value_ptr(projectionMatrix));

    glBufferSubData(GL_ARRAY_BUFFER,
                     0,
                     particles->size() * sizeof(Particle),
                     particles->data());

    glDrawArrays(GL_POINTS, 0, particles->size());

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    glfwPollEvents();
}
void Renderer::prepareImGuiFrame() {
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 10.0f, 10.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::Begin("Konfiguracja symulacji", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

    // ##### CONFIG #####

    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Zmienne konfiguracyjne:");
    ImGui::InputFloat("Mnoznik G", &G_MULTIPLIER, 1.0f, 10.0f, "%.1f");

    int power = 0;
    while ((int)std::pow(2, power) < SPLIT_AT_LEAF_SIZE && power < 13) {
        power++;
    }
    char format_buf[32];
    snprintf(format_buf, sizeof(format_buf), "%d", 1 << power);
    if (ImGui::SliderInt("Podzial drezwa przy", &power, 0, 13, format_buf)) {
        SPLIT_AT_LEAF_SIZE = (int)std::pow(2, power);
    }

    if (ImGui::SliderFloat("Theta", &THETA, 0.0f, 5.0f)) THETA_SQ = THETA * THETA;
    if (ImGui::SliderFloat("Epsilon", &EPSILON, 0.01f, 5.0f)) EPSILON_SQ = EPSILON * EPSILON;
    ImGui::InputFloat("Krok czasowy", &TIME_STEP, 10.0f, 1000.0f, "%.1f");
    ImGui::SliderInt("Watki", &NUM_THREADS, 1, MAX_HARDWARE_THREADS);
    ImGui::Separator();

    // ##### CONFIG #####
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Dane profilowe:");
    ImGui::Text("TPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Liczba cial: %zu", particles->size());
    ImGui::Text("Wierzcholki: %d", octree->nodeCount);
    ImGui::Text("Interakcje COM: %d", COM_INTERACTIONS);
    ImGui::Text("Bezposrednie interakcje: %d", DIRECT_INTERACTIONS);
    ImGui::Checkbox("Licz interakcje", &countInteractions);
    ImGui::Separator();

    // ##### PARTICLE GENERATOR #####
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Generator czastek:");
    if (ImGui::Button("Usun ciala", ImVec2(-1, 0))) particles->clear();
    ImGui::InputFloat("Min. promien", &minRadius, 100.0f, 1000.0f, "%.1f");
    ImGui::InputFloat("Maks. promien", &maxRadius, 100.0f, 1000.0f, "%.1f");
    ImGui::InputFloat("Masa czastki", &genParticleMass, 100.0f, 1000.0f, "%.1f");
    ImGui::InputFloat("Masa centralna", &genCenterMass, 1000.0f, 10000.0f, "%.1f");
    ImGui::InputInt("Liczba czastek", &genCount, 1000, 10000);
    ImGui::Checkbox("Zakotwicz", &ANCHOR);

    glm::vec3 FOC = camera.position + camera.viewDirection * 150.0f;
    glm::vec3 CVV = camera.currentVelocity * (deltaTime / std::max(0.0001f, TIME_STEP));

    if (ImGui::Button("Stworz czastke", ImVec2(-1, 0))) {
        ParticleGenerator::addParticle(*particles, FOC.x, FOC.y, FOC.z, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Stworz prostokat", ImVec2(-1, 0))) {
        ParticleGenerator::createFlatRectangle(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Stworz szescian", ImVec2(-1, 0))) {
        ParticleGenerator::createCube(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Stworz dysk", ImVec2(-1, 0))) {
        ParticleGenerator::createDisc(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, genCenterMass, minRadius, maxRadius, CVV.x, CVV.y, CVV.z);
    }
    if (ImGui::Button("Stworz kule", ImVec2(-1, 0))) {
        ParticleGenerator::createSphere(*particles, FOC.x, FOC.y, FOC.z, genCount, genParticleMass, genCenterMass, minRadius, maxRadius, CVV.x, CVV.y, CVV.z);
    }
    ImGui::Separator();
    // ##### PARTICLE GENERATOR #####


    // ##### CAMERA #####
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Kamera:");
    ImGui::Text("Predkosc: %.1f", camera.speed);
    ImGui::Text("Pochylenie: %.1f", camera.pitch);
    ImGui::Text("Odchylenie: %.1f", camera.yaw);
    ImGui::Text("Pozycja: %.2f, %.2f, %.2f", camera.position.x, camera.position.y, camera.position.z);
    if (ImGui::Button("Zresetuj pozycje", ImVec2(-1, 0))) camera.position = glm::vec3(0,0,0);
    // ##### CAMERA #####

    ImGui::End();
}




