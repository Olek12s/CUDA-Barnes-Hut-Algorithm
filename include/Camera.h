//
// Created by Oleki on 08.06.2026.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera();
    Camera(float x, float y, float z);
    void update(GLFWwindow* window, float deltaTime);
    void keyboardInput(GLFWwindow *window, float deltaTime);
    void mouseInput(float x, float y);
    glm::vec3 getUpDirection();
    glm::vec3 getRightDirection();
    glm::mat4 getViewMatrix();
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    float pitch;
    float yaw;
    float lastX;
    float lastY;
    bool mouseMoved;
    float speed;
    glm::vec3 currentVelocity;
    float fov;
    glm::vec3 position;
    glm::vec3 viewDirection;
    glm::mat4 view;

private:
    bool lastAddPressed = false;
    bool lastSubPressed = false;
};


#endif //CAMERA_H
