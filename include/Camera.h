//
// Created by Oleki on 08.06.2026.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Camera {
public:
    Camera();   // init with default position (0,0,0)
    Camera(float x, float y, float z);
    void update(GLFWwindow* window);

private:
    glm::vec3 getUpDirection();
    glm::vec3 getRightDirection();

    glm::vec3 position;         // position of camera in world
    glm::vec3 viewDirection;    // direction of looking, default (1,0,0)
    glm::mat4 view;             // View matrix which transforms world space into camera (view) space
};


#endif //CAMERA_H
