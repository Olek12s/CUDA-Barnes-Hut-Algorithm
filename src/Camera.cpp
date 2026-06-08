//
// Created by Oleki on 08.06.2026.
//

#include "Camera.h"




Camera::Camera() {
    position = glm::vec3(0,0,3);
    viewDirection = glm::vec3(0,0,-1);
    fov = 90.0f;

    pitch = 0.f;
    yaw = -90.0f;

    lastX = 400;
    lastY = 300;
    mouseMoved = true;
    speed = 0.1f;
}

Camera::Camera(float x, float y, float z) {
    position = glm::vec3(x,y,z);
}

glm::mat4 Camera::getViewMatrix()
{
    return glm::lookAt(position, position + viewDirection, glm::vec3(0,1,0));
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const
{
    return glm::perspective(glm::radians(fov), aspectRatio, 0.01f, 10000000.0f);
}

glm::vec3 Camera::getRightDirection() {
    return glm::vec3(glm::normalize(glm::cross(glm::vec3(0, 1, 0), viewDirection)));
}

glm::vec3 Camera::getUpDirection() {
    return glm::vec3(glm::cross(viewDirection, getRightDirection()));
}

void Camera::update(GLFWwindow *window) {
    // ##### Handle inputs #####
    keyboardInput(window);
}

void Camera::keyboardInput(GLFWwindow *window) {
    auto pressed = GLFW_PRESS;

    glm::vec3 forward = viewDirection;
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0,1,0)));
    glm::vec3 up = glm::vec3(0,1,0);

    if (glfwGetKey(window, GLFW_KEY_W) == pressed || glfwGetKey(window, GLFW_KEY_UP) == pressed) {
        position += speed * forward;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == pressed || glfwGetKey(window, GLFW_KEY_LEFT) == pressed) {
        position += speed * -right;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == pressed || glfwGetKey(window, GLFW_KEY_DOWN) == pressed) {
        position += speed * -forward;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == pressed || glfwGetKey(window, GLFW_KEY_RIGHT) == pressed) {
        position += speed * right;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == pressed) {
        position += speed * -up;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == pressed) {
        position += speed * up;
    }
    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == pressed) {
        speed *= 2;
    }
    if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == pressed) {
        speed /= 2;
    }
}

void Camera::mouseInput(float x, float y) {
    if (mouseMoved)
    {
        lastX = x;
        lastY = y;
        mouseMoved = false;
    }

    // offset between last frame vs current frame of how many pixels mouse has moved
    float xoffset = x - lastX;
    float yoffset = lastY - y;

    lastX = x;
    lastY = y;

    float sensitivity = 0.1f;   //TODO:
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    // update camera's pitch & yaw variables after mouse movement
    pitch += yoffset;
    yaw += xoffset;

    // lock the camera's angles (at 90 degrees there's  LookAt camera flip)
    if (pitch > 89.f) pitch = 89.9f;
    if (pitch < -89.f) pitch = -89.9f;

    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    viewDirection = glm::normalize(dir);
}
