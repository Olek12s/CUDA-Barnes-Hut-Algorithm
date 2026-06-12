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

#include "Globals.h"
#include "Octree.h"
#include "Renderer.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

int glTest();
void cudaApiTest();

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
    std::uniform_real_distribution<float> dist(-1000.0f, 1.0f);

    for (size_t i = 0; i < n; i++)
    {
        float x = dist(gen);
        float y = dist(gen);
        float z = dist(gen);
        float mass = 100;

        Particle p{x, y, z, mass}; // zero-initialization (IMPORTANT)

        particles.push_back(p);
    }



    particles.push_back(Particle(1.0, 100.0 , 0.5, 1.f));
    particles.push_back(Particle(1.0, 105.0 , 0.5, 1500000.f));
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

    size_t n = 000'000;
    std::vector<Particle> particles;

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

    Octree octtree;
    Renderer renderer(particles, octtree);
    renderer.init();

    std::array<double, 11> timings = {0.0};

    auto fpsTimer = std::chrono::steady_clock::now();
    int frameCount = 0;


    while (!renderer.isTerminated) {
        // 1. render
        auto t0 = std::chrono::high_resolution_clock::now();
        //renderer.frameTick();
        renderer.initFrame();
        timings[0] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 2. integrate w/ leapfrog (velocity step 1/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            //std::cout << "mass=" << p.mass << " ax=" << p.ax << "\n";
            p.leapFrogVelStep(TIME_STEP * 0.5f);
        }
        timings[1] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 2.5 integrate w/ leapfrog (position step)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogPosStep(TIME_STEP);
        }
        timings[2] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 3. bounds
        t0 = std::chrono::high_resolution_clock::now();
        auto bounds = findMinMax(particles);
        timings[3] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 4. recompute morton codes
        t0 = std::chrono::high_resolution_clock::now();
        computeMortonCodes(particles, bounds);
        timings[4] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 5. sort by morton
        t0 = std::chrono::high_resolution_clock::now();
        std::sort(particles.begin(), particles.end(), comp);
        timings[5] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 6. rebuild tree
        t0 = std::chrono::high_resolution_clock::now();
        octtree.buildTree(particles);
        timings[6] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 7. mass distribution
        t0 = std::chrono::high_resolution_clock::now();
        octtree.computeMassDistribution(particles);
        timings[7] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 8. reset accelerations
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.ax = p.ay = p.az = 0;
        }
        timings[8] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 9. compute forces (multithread)
        t0 = std::chrono::high_resolution_clock::now();


        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);

        auto worker = [&](size_t start, size_t end)
        {
            for (size_t i = start; i < end; i++)
            {
                octtree.computeForcesAffectingParticle(0, particles[i], particles);
            }
        };

        size_t n = particles.size();
        size_t chunk = (n + NUM_THREADS - 1) / NUM_THREADS;

        for (unsigned int t = 0; t < NUM_THREADS; t++)
        {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, n);

            threads.emplace_back(worker, start, end);
        }

        for (auto &th : threads)
        {
            th.join();
        }

        timings[9] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0
        ).count();

        // 10. integrate w/ leapfrog (velocity step 2/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogVelStep(TIME_STEP * 0.5f);
            //p.leapFrogPosStep(TIME_STEP);  posstep should appear twice
        }
        timings[10] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        renderer.prepareImGuiFrame();   // prepare imgui window
        renderer.renderFrame();         // render particles

        frameCount++;
        auto now = std::chrono::steady_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsTimer).count();

        if (elapsed >= 1000)    // once per second
        {
            double fps = frameCount * 1000.0 / elapsed;
            std::cout << "\nCAM pos: [" << renderer.camera.position.x << ", " << renderer.camera.position.y << ", " << renderer.camera.position.z << "]\n";
            std::cout << "FPS: " << fps << '\n';
            std::cout << "Nodes: " << octtree.nodeCount << "\n";

            fpsTimer = now;
            double totalTime = 0.0;
            for (double t : timings) {
                totalTime += t;
            }

            const char* names[11] =
            {
                "1. render",
                "2. leapfrog vel step 1/2",
                "3. leapfrog pos step",
                "4. bounds",
                "5. morton codes",
                "6. sort morton",
                "7. build tree",
                "8. mass distribution",
                "9. reset accelerations",
                "10. compute forces",
                "11. leapfrog vel step 2/2"
            };

            std::cout << "\n===== PROFILING FOR " << particles.size() << " BODIES (LAST FRAME) =====\n";

            for (int i = 0; i < 11; ++i)
            {
                double percent = (timings[i] / totalTime) * 100.0;

                std::cout
                    << names[i]
                    << ": "
                    << timings[i]
                    << " ms ("
                    << percent
                    << "%)\n";
            }
            std::cout << "TOTAL frame: " << totalTime << " ms\n";
            frameCount = 0;
        }
    }

    //cudaApiTest();
    //glTest();
    return 0;
}
