//
// Created by Oleki on 08.06.2026.
//

#include "Camera.h"

Camera::Camera() {
    position = glm::vec3(0,0,3);
    viewDirection = glm::vec3(0,0,-1);
    fov = 110.0f;

    pitch = 0.f;
    yaw = -90.0f;

    lastX = 0;
    lastY = 0;
    mouseMoved = true;
    speed = 1000.f;
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
    //return glm::perspective(glm::radians(fov), aspectRatio, 0.01f, 10000000.0f);
    return glm::infinitePerspective(glm::radians(fov), aspectRatio, 0.01f);
}

glm::vec3 Camera::getRightDirection() {
    return glm::vec3(glm::normalize(glm::cross(glm::vec3(0, 1, 0), viewDirection)));
}

glm::vec3 Camera::getUpDirection() {
    return glm::vec3(glm::cross(viewDirection, getRightDirection()));
}

void Camera::update(GLFWwindow *window, float deltaTime) {
    keyboardInput(window, deltaTime);
}

void Camera::keyboardInput(GLFWwindow *window, float deltaTime) {
    auto pressed = GLFW_PRESS;

    glm::vec3 forward = viewDirection;
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0,1,0)));
    glm::vec3 up = glm::vec3(0,1,0);

    glm::vec3 moveDirection(0.0f);

    if (glfwGetKey(window, GLFW_KEY_W) == pressed || glfwGetKey(window, GLFW_KEY_UP) == pressed) {
        moveDirection += forward;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == pressed || glfwGetKey(window, GLFW_KEY_LEFT) == pressed) {
        moveDirection += -right;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == pressed || glfwGetKey(window, GLFW_KEY_DOWN) == pressed) {
        moveDirection += -forward;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == pressed || glfwGetKey(window, GLFW_KEY_RIGHT) == pressed) {
        moveDirection += right;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == pressed) {
        moveDirection += -up;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == pressed) {
        moveDirection += up;
    }

    if (glm::length(moveDirection) > 0.0f) {
        currentVelocity = glm::normalize(moveDirection) * speed;
        position += currentVelocity * deltaTime;
    } else {
        currentVelocity = glm::vec3(0.0f);
    }

    bool addPressed = (glfwGetKey(window, GLFW_KEY_KP_ADD) == pressed || glfwGetKey(window, GLFW_KEY_EQUAL) == pressed);
    if (addPressed && !lastAddPressed) {
        speed *= 2.0f;
    }
    lastAddPressed = addPressed;

    bool subPressed = (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == pressed || glfwGetKey(window, GLFW_KEY_MINUS) == pressed);
    if (subPressed && !lastSubPressed) {
        speed /= 2.0f;
    }
    lastSubPressed = subPressed;
}

void Camera::mouseInput(float x, float y) {
    if (mouseMoved)
    {
        lastX = x;
        lastY = y;
        mouseMoved = false;
    }

    float xoffset = x - lastX;
    float yoffset = lastY - y;
    lastX = x;
    lastY = y;

    xoffset *= 0.1f;
    yoffset *= 0.1f;

    pitch += yoffset;
    yaw += xoffset;

    if (yaw > 360.0f) yaw -= 360.0f;
    if (yaw < -360.0f) yaw += 360.0f;
    if (pitch > 89.f) pitch = 89.f;
    if (pitch < -89.f) pitch = -89.f;

    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    viewDirection = glm::normalize(dir);
}
