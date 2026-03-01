#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <array>
#include <utility>

#include "cuda.cuh"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

int glTest();
void cudaApiTest();


struct Z_CODE {
    unsigned int x_bits;
    unsigned int y_bits;
    unsigned int z_bits;
};

struct Particle {
    float x, y, z;
};

// scale float value to new value between [0, UINT_MAX]
unsigned int scale(float f, float fmin, float fmax) {
    float clamped = (f - fmin) / (fmax - fmin); // [0,1]
    if(clamped < 0.f) clamped = 0.f;
    if(clamped > 1.f) clamped = 1.f;
    return (unsigned int)(clamped * 0xFFFFFFFF);
}

std::array<std::pair<float,float>, 3> findMinMax(std::vector<Particle>& particles) {
    std::array<std::pair<float, float>, 3> bounds =
    {{
        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::lowest()},

        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::lowest()},

        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::lowest()}
    }};

    for (auto &p : particles) {
        // x
        bounds[0].first = std::min(bounds[0].first, p.x);
        bounds[0].second = std::max(bounds[0].second, p.x);

        // y
        bounds[1].first = std::min(bounds[1].first, p.y);
        bounds[1].second = std::max(bounds[1].second, p.y);

        // z
        bounds[2].first = std::min(bounds[2].first, p.z);
        bounds[2].second = std::max(bounds[2].second, p.z);
    }
}

Z_CODE interlace(const Particle& p,float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
    Z_CODE code;
    code.x_bits = scale(p.x, xmin, xmax);
    code.y_bits = scale(p.y, ymin, ymax);
    code.z_bits = scale(p.z, zmin, zmax);
}

// true if a > b
// false if a < b
bool compareZOrder(const Particle &a, const Particle &b) {
    Z_CODE Za = interlace(a, xmin, xmax, ymin, ymax, zmin, zmax);
    Z_CODE Zb = interlace(b, xmin, xmax, ymin, ymax, zmin, zmax);

    if (Za.x_bits != Zb.x_bits) return Za.x_bits < Zb.x_bits;
    if (Za.y_bits != Zb.y_bits) return Za.y_bits < Zb.y_bits;
    return Za.z_bits < Zb.z_bits;
}

int main() {

    std::cout << "test11\n";


    std::vector<Particle> particles = {
        {0.1f, 0.5f, 0.7f},
            {0.4f, 0.2f, 0.9f},
            {0.8f, 0.3f, 0.1f}
    };

    std::sort(particles.begin(), particles.end(), compareZOrder);

    for (auto &p : particles) {
        std::cout << p.x << " " << p.y << " " << p.z << '\n';
    }


    //cudaApiTest();
    //glTest();
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
