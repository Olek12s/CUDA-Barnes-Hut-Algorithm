#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <array>
#include <random>
#include <utility>

#include "cuda.cuh"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

int glTest();
void cudaApiTest();

constexpr int MAX_MORTON_BITS = 21; // Z_CODE has 64 unsigned bit type - code is defined by 3 values, thus maximum morton bits is 64/3 = 21
constexpr unsigned int MORTON_SCALE = (1u << MAX_MORTON_BITS) - 1u;

struct Particle {
    float x, y, z;
    uint64_t Z_CODE;
};

// scale float value to new value between [0, UINT_MAX]
unsigned int scale(float f, float fmin, float fmax) {

    float clamped = (f - fmin) / (fmax - fmin); // [0,1]

    if(clamped < 0.f) clamped = 0.f;
    if(clamped > 1.f) clamped = 1.f;
    return (unsigned int)(clamped * MORTON_SCALE);
}


// expand method could be replaced with naive method iterating through every of 21 bits and doing 3 operations:
//
//      morton |= ((x >> i) & 1ull) << (3*i);
//      morton |= ((y >> i) & 1ull) << (3*i+1);
//      morton |= ((z >> i) & 1ull) << (3*i+2);
//
//  But above method is multiple times slower than the one below.

uint64_t expand(unsigned int v) {
    uint64_t x = v & 0x1fffff;
    // 21 bits       // 0b00000000 00000000 00000000 00000000 00000000 00011111 11111111 11111111

    // initial v example value: abcd

    // spacing: 32
    x = (x | x << 32) & 0x1f00000000ffff;       // 0b00000000 00011111 00000000 00000000 00000000 00000000 11111111 11111111

    // spacing: 16
    x = (x | x << 16) & 0x1f0000ff0000ff;       // 0b00000000 00011111 00000000 00000000 11111111 00000000 00000000 11111111

    // spacing: 8
    x = (x | x << 8)  & 0x100f00f00f00f00f;     // 0b00010000 00001111 00000000 11110000 00001111 00000000 11110000 00001111

    // a0b0c0d0         spacing: 2
    x = (x | x << 4)  & 0x10c30c30c30c30c3;     // 0b00010000 11000011 00001100 00110000 11000011 00001100 00110000 11000011

    // a00b00c00d00     spacing: 3
    x = (x | x << 2)  & 0x1249249249249249;     // 0b00010010 01001001 00100100 10010010 01001001 00100100 10010010 01001001


    return x;
}

uint64_t getMortonCodeFrom3D(float x, float y, float z, const std::array<std::pair<float,float>,3>& bounds) {
    uint32_t xs = scale(x, bounds[0].first, bounds[0].second);
    uint32_t ys = scale(y, bounds[1].first, bounds[1].second);
    uint32_t zs = scale(z, bounds[2].first, bounds[2].second);

    uint64_t xx = expand(xs);
    uint64_t yy = expand(ys);
    uint64_t zz = expand(zs);

    // interlace (x,y,z) bits in pattern: x₀y₀z₀x₁y₁z₁ [...]
    return xx | (yy << 1) | (zz << 2);
}

void computeMortonCodes(std::vector<Particle>& particles,const std::array<std::pair<float,float>,3>& bounds)
{
    for(auto& p : particles)
    {
        p.Z_CODE = getMortonCodeFrom3D(p.x, p.y, p.z, bounds);
    }
}

bool comp(const Particle& a, const Particle& b)
{
    return a.Z_CODE < b.Z_CODE;
}


// find boundary float values of particles vector
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

    return bounds;
}

std::vector<Particle> generateParticles(size_t n)
{
    std::vector<Particle> particles;
    particles.reserve(n);

    std::random_device rd;
    std::mt19937 gen(rd());

    // [-1000, 1000]
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (size_t i = 0; i < n; i++)
    {
        particles.push_back({dist(gen),dist(gen),dist(gen)});
    }

    float limits[2] = { -1000, 1000 };
    for (float x : limits)
    {
        for (float y : limits)
        {
            for (float z : limits)
            {
                particles.push_back({x, y, z});
            }
        }
    }

    return particles;
}

int main() {
    std::cout << "test11\n";

    // std::vector<Particle> particles = {
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, -400.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {300.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 222.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 222.5f, 0.7f}, {0.4f, 0.2f, -8880.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {0.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    //     {500.1f, 0.5f, 0.7f}, {0.4f, 0.2f, 0.9f},{0.8f, 0.3f, 0.1f}, {0.5, 0.2, -0.1},
    // };

    size_t n = 1000;
    std::vector<Particle> particles = generateParticles(n);

    auto bounds = findMinMax(particles);
    computeMortonCodes(particles, bounds);
    std::sort(particles.begin(), particles.end(), comp);

    for (auto &p : particles) {
        std::cout << p.x << " " << p.y << " " << p.z << '\n';
    }

    bool monotone = true;
    for(size_t i = 1; i < particles.size(); i++)
    {
        if(particles[i].Z_CODE < particles[i-1].Z_CODE)
        {
            monotone = false;
            break;
        }
    }
    std::cout << "Monotonic Z_CODE: " << (monotone ? "OK" : "FAIL") << "\n\n\n";

    for (auto &p : particles) {
        std::cout << p.Z_CODE << "\n";
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
