#include <iostream>
#include <vector>

#include "cuda.cuh"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

int glTest();
void cudaApiTest();

int main() {

    std::cout << "test6\n";

    cudaApiTest();
    glTest();
    return 0;
}

void cudaApiTest() {

    int n = 1000;
    std::vector<float> arr(n, 0.0f);

    cuda_sort_example(arr.data(), n);
    cudaTest();
}

int glTest()
{
    // ======================================================
// OPENGL TEST TRIANGLE
// ======================================================

std::cout << "\nStarting OpenGL test...\n";

if (!glfwInit())
{
    std::cout << "GLFW init failed\n";
    return -1;
}

glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

GLFWwindow* window =
    glfwCreateWindow(800, 600, "OpenGL Test", nullptr, nullptr);

if (!window)
{
    std::cout << "Window creation failed\n";
    glfwTerminate();
    return -1;
}

glfwMakeContextCurrent(window);

if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
{
    std::cout << "GLAD init failed\n";
    return -1;
}

std::cout << "OpenGL OK: "
          << glGetString(GL_VERSION) << std::endl;

// ================== TRIANGLE DATA ==================

float vertices[] =
{
     0.0f,  0.5f, 0.0f,
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f
};

// ================== SHADERS ==================

const char* vertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main(){ gl_Position = vec4(aPos,1.0); }";

const char* fragmentShaderSource =
"#version 330 core\n"
"out vec4 FragColor;\n"
"void main(){ FragColor = vec4(0.2,0.6,1.0,1.0); }";

// vertex shader
GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vShader, 1, &vertexShaderSource, nullptr);
glCompileShader(vShader);

// fragment shader
GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fShader, 1, &fragmentShaderSource, nullptr);
glCompileShader(fShader);

// program
GLuint shaderProgram = glCreateProgram();
glAttachShader(shaderProgram, vShader);
glAttachShader(shaderProgram, fShader);
glLinkProgram(shaderProgram);

glDeleteShader(vShader);
glDeleteShader(fShader);

// ================== BUFFERS ==================

GLuint VAO, VBO;

glGenVertexArrays(1, &VAO);
glGenBuffers(1, &VBO);

glBindVertexArray(VAO);

glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER,
             sizeof(vertices),
             vertices,
             GL_STATIC_DRAW);

glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
glEnableVertexAttribArray(0);

// ================== RENDER LOOP ==================

while(!glfwWindowShouldClose(window))
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    glClearColor(0.05f,0.05f,0.1f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);

    glDrawArrays(GL_TRIANGLES,0,3);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

glfwTerminate();

    return 0;
}
