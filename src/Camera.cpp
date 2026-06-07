//
// Created by Oleki on 08.06.2026.
//

#include "Camera.h"




Camera::Camera() {
    position = glm::vec3(0,0,0);
}

Camera::Camera(float x, float y, float z) {
    position = glm::vec3(x,y,z);
}

glm::vec3 Camera::getRightDirection() {
    return glm::vec3(glm::normalize(glm::cross(glm::vec3(0, 1, 0), viewDirection)));
}

glm::vec3 Camera::getUpDirection() {
    return glm::vec3(glm::cross(viewDirection, getRightDirection()));
}

void Camera::update(GLFWwindow *window) {
    auto pressed = GLFW_PRESS;
    float speed = 0.01f;
    if (glfwGetKey(window, GLFW_KEY_W) == pressed || glfwGetKey(window, GLFW_KEY_UP) == pressed) {
        std::cout << "W";
        position += speed * glm::vec3(0, 0, -1);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == pressed || glfwGetKey(window, GLFW_KEY_LEFT) == pressed) {
        std::cout << "A";
        position += speed * glm::vec3(-1, 0, 0);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == pressed || glfwGetKey(window, GLFW_KEY_DOWN) == pressed) {
        std::cout << "S";
        position += speed * glm::vec3(0, 0, 1);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == pressed || glfwGetKey(window, GLFW_KEY_RIGHT) == pressed) {
        std::cout << "D";
        position += speed * glm::vec3(1, 0, 0);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
        std::cout << "SHIFT";
        position += speed * glm::vec3(0, -1, 0);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE)) {
        std::cout << "SPACE";
        position += speed * glm::vec3(0, 1, 0);
    }
}