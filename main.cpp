#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <array>
#include <bitset>
#include <random>
#include <utility>
#include <thread>
#include <chrono>

#include "Octtree.h"
#include "Renderer.h"
#include "src/cuda/cuda.cuh"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

int glTest();
void cudaApiTest();

constexpr int MAX_MORTON_BITS = 21; // Z_CODE has 64 unsigned bit type - code is defined by 3 values, thus maximum morton bits is 64/3 = 21
constexpr unsigned int MORTON_SCALE = (1u << MAX_MORTON_BITS) - 1u; // 2097151, or std::pow(2, 21
                                                                    //Morton_SCALE is in other words the biggest digit possible toencode on 21 btis

// scale float value to new value between [0, 21bits]
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


//x = x2 x1 x0
//y = y2 y1 y0
//z = z2 z1 z0
//res: z2 y2 x2 z1 y1 x1 z0 y0 x0
uint64_t getMortonCodeFrom3D(float x, float y, float z, const std::array<std::pair<float,float>,3>& bounds) {
    uint32_t xs = scale(x, bounds[0].first, bounds[0].second);
    uint32_t ys = scale(y, bounds[1].first, bounds[1].second);
    uint32_t zs = scale(z, bounds[2].first, bounds[2].second);

    uint64_t xx = expand(xs);
    uint64_t yy = expand(ys);
    uint64_t zz = expand(zs);

    // interlace (x,y,z) bits in pattern: [...]x₁y₁z₁x₀y₀z₀. This value determines tree's octant. Octant = (z,y,x)

    // Division values for 8 octants at the same level:
    // 000 - back-bottom-left
    // 001 - back-bottom-right
    // 010 - back-top-left
    // 011 - back-top-right
    // 100 - front-bottom-left
    // 101 - front-bottom-right
    // 110 - front-top-left
    // 111 - front-top-right

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
    std::uniform_real_distribution<float> dist(-100000.0f, 100000.0f);

    for (size_t i = 0; i < n; i++)
    {
        Particle p{}; // zero-initialization (IMPORTANT)

        p.x = dist(gen);
        p.y = dist(gen);
        p.z = dist(gen);
        p.mass = 1;

        particles.push_back(p);
    }
    particles.push_back(Particle(0.3, 0.5 , 0));
    return particles;
}

void buildTree(std::vector<Particle> &sortedParticles) {

}

int main() {

    // Packets for tree are distributed as:

    // Bounding box: [-1000, 1000] along each axis
    // Level 1 (root octant, first Morton code “package”):
    //   - [-1000, 0)   // negative half
    //   - [0, 1000]    // positive half

    // Level 2 (second Morton code package, divide each half of level 1 in half):
    //   - [-1000, -500)
    //   - [-500, 0)
    //   - [0, 500)
    //   - [500, 1000]

    // Level 3 (third Morton code package, divide each previous quarter in half):
    //   - [-1000, -750)
    //   - [-750, -500)
    //   - [-500, -250)
    //   - [-250, 0)
    //   - [0, 250)
    //   - [250, 500)
    //   - [500, 750)
    //   - [750, 1000]

    // Level 4 (fourth package, node size = 2000 / 16 = 125):
    //   - [-1000, -875), [-875, -750), [-750, -625), ..., [875, 1000]

    // Level 5 (fifth package, node size = 62.5):
    //   - Each previous interval is halved again
    //   - Example: [-1000, -937.5), [-937.5, -875), [-875, -812.5), ...

    // ...

    // Level 21 (last package, node size = 2000 / 2^21 ~ 0.00095):
    //   - Each Morton code package corresponds to a single leaf node

    // std::vector<Particle> particles = {
    //     // bounds
    //     {-1000, -1000, -1000},
    //      {-1000, -1000, 1000},
    //     // {-1000, 1000, -1000},
    //     // {-1000, 1000, 1000},
    //     // {1000, -1000, 1000},
    //     // {1000, -1000, -1000},
    //     // {1000, 1000, -1000},
    //     // {1000, 1000, 1000},
    //
    //     // 000 octant 0: x ∈ [-1000, 0) y ∈ [-1000, 0) z ∈ [-1000, 0)
    //     //{-50.f, -760.f, -50.f},
    //     // {-300.f, -300.f, -300.f},
    //     // {-550.f, -550.f, -550.f},
    //     // {-800.f, -800.f, -800.f},
    //
    //     // // 001 octant 1: x ∈ [0, 1000] y ∈ [-1000, 0) z ∈ [-1000, 0)
    //     // {50.f, -50.f, -50.f},
    //     // {300.f, -300.f, -300.f},
    //     // {550.f, -550.f, -550.f},
    //     // {800.f, -800.f, -800.f},
    //     //
    //     // // 010 octant 2: x ∈ [-1000, 0) y ∈ [0, 1000] z ∈ [-1000, 0)
    //     // {-50.f, 50.f, -50.f},
    //     // {-300.f, 300.f, -300.f},
    //     // {-550.f, 550.f, -550.f},
    //     // {-800.f, 800.f, -800.f},
    //     //
    //     // // 011 octant 3 x ∈ [0, 1000] y ∈ [0, 1000] z ∈ [-1000, 0)
    //     // {50.f, 50.f, -50.f},
    //     // {300.f, 300.f, -300.f},
    //     // {550.f, 550.f, -550.f},
    //     // {800.f, 800.f, -800.f},
    //     //
    //     // // 100 octant 4 x ∈ [-1000, 0) y ∈ [-1000, 0) z ∈ [0, 1000]
    //     // {-50.f, -50.f, 50.f},
    //     // {-300.f, -300.f, 300.f},
    //     // {-550.f, -550.f, 550.f},
    //     // {-800.f, -800.f, 800.f},
    //     //
    //     // // 101 octant 5 x ∈ [0, 1000] y ∈ [-1000, 0) z ∈ [0, 1000]
    //     // {50.f, -50.f, 50.f},
    //     // {300.f, -300.f, 300.f},
    //     // {550.f, -550.f, 550.f},
    //     // {800.f, -800.f, 800.f},
    //     //
    //     // // 110 octant 6 x ∈ [-1000, 0) y ∈ [0, 1000] z ∈ [0, 1000]
    //     // {-50.f, 50.f, 50.f},
    //     // {-300.f, 300.f, 300.f},
    //     // {-550.f, 550.f, 550.f},
    //     // {-800.f, 800.f, 800.f},
    //     //
    //     // // 111 octant 7 x ∈ [0, 1000] y ∈ [0, 1000] z ∈ [0, 1000]
    //     // {50.f, 50.f, 50.f},
    //     // {300.f, 300.f, 300.f},
    //     // {550.f, 550.f, 550.f},
    //     // {800.f, 800.f, 800.f},
    // };

    size_t n = 0;
    std::vector<Particle> particles = generateParticles(n);

    auto bounds = findMinMax(particles);
    computeMortonCodes(particles, bounds);
    std::sort(particles.begin(), particles.end(), comp);

    bool monotone = true;
    for(size_t i = 1; i < particles.size(); i++)
    {
        if(particles[i].Z_CODE < particles[i-1].Z_CODE)
        {
            monotone = false;
            break;
        }
    }

    for (auto &p : particles) {
        //std::cout << p.Z_CODE << "\n";
       // std::cout << std::bitset<64>(p.Z_CODE) << "\n";
       // std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")    ";
        std::bitset<64> bits(p.Z_CODE);
        for (int i = 63; i >= 0; --i)
        {
          //  std::cout << bits[i];
          //  if (i == 63 || (i % 3 == 0 && i != 0))std::cout << ' ';
        }
      //  std::cout << '\n';
    }
   // std::cout << "Monotonic Z_CODE: " << (monotone ? "OK" : "FAIL") << "\n\n\n";

    Renderer renderer(particles);
    renderer.init();
    Octtree octtree;

    float timeStep = 10.f;
    while (!renderer.isTerminated) {
        renderer.frameTick();

        // 1. integrate w/ leapfrog (velocity step 1/2)
        for (auto &p : particles) {
            std::cout << p.toString() << "\n";
            p.leapFrogVelStep(timeStep * 0.5f);
        }
        // 1.5 integrate w/ leapfrog (position step)
        for (auto &p : particles) {
            p.leapFrogPosStep(timeStep);
        }

        // 2. bounds
        auto bounds = findMinMax(particles);

        // 3. recompute morton codes
        computeMortonCodes(particles, bounds);

        // 4. sort by morton
        std::sort(particles.begin(), particles.end(), comp);

        // 5. rebuild tree
        octtree.buildTree(particles);

        //
        octtree.computeMassDistribution(particles);
        // 6. reset accelerations
        for (auto &p : particles) {
            p.ax = p.ay = p.az = 0;
        }

        // 7. compute forces
        for (auto &p : particles) {
            octtree.computeForcesAffectingParticle(0, p, p.ax, p.ay, p.az, particles);
        }

        // 6. integrate w/ leapfrog (velocity step 2/2)
        for (auto &p : particles) {
            p.leapFrogVelStep(timeStep * 0.5f);
            p.leapFrogPosStep(timeStep);
        }

        Particle p = particles[0];
        //std::cout << p.x << "\n";
        //std::cout << p.toString() << "\n";
        //printf("posX: %.9f\n", p.x);

        // std::this_thread::sleep_for(
        //       std::chrono::milliseconds(16)
        //   );

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        std::cout << "." << std::flush;

    }

    //cudaApiTest();
    //glTest();
    return 0;
}
