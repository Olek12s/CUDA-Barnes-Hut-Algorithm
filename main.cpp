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

unsigned int scale(float f, float fmin, float fmax) {

    float clamped = (f - fmin) / (fmax - fmin);

    if(clamped < 0.f) clamped = 0.f;
    if(clamped > 1.f) clamped = 1.f;
    return (unsigned int)(clamped * MORTON_SCALE);
}

uint64_t getMortonCodeFrom3D(float x, float y, float z, const std::array<std::pair<float,float>,3>& bounds) {
    // scale
    uint64_t xs = scale(x, bounds[0].first, bounds[0].second);
    uint64_t ys = scale(y, bounds[1].first, bounds[1].second);
    uint64_t zs = scale(z, bounds[2].first, bounds[2].second);

    uint64_t morton = 0;

    for (int i = 0; i < 21; i++) {
        morton |= ((xs >> i) & 1ull) << (3 * i);
        morton |= ((ys >> i) & 1ull) << (3 * i + 1);
        morton |= ((zs >> i) & 1ull) << (3 * i + 2);
    }

    return morton;
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

int main() {
    std::vector<Particle> particles;
    Octree octtree;
    Renderer renderer(particles, octtree);
    renderer.init();

    std::array<double, 11> accumulatedTimings = {0.0};
    auto tpsTimer = std::chrono::steady_clock::now();
    int frameCount = 0;

    while (!renderer.isTerminated) {
        // 1. render
        auto t0 = std::chrono::high_resolution_clock::now();
        renderer.initFrame();
        renderer.prepareImGuiFrame();
        renderer.renderFrame();         // render
        frameCount++;
        accumulatedTimings[0] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 2. integrate w/ leapfrog (velocity step 1/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogVelStep(TIME_STEP * 0.5f);
        }
        accumulatedTimings[1] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 2.5 integrate w/ leapfrog (position step)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogPosStep(TIME_STEP);
        }
        accumulatedTimings[2] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 3. bounds
        t0 = std::chrono::high_resolution_clock::now();
        auto bounds = findMinMax(particles);
        accumulatedTimings[3] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 4. recompute morton codes
        t0 = std::chrono::high_resolution_clock::now();
        computeMortonCodes(particles, bounds);
        accumulatedTimings[4] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 5. sort by morton
        t0 = std::chrono::high_resolution_clock::now();
        std::sort(particles.begin(), particles.end(), comp);
        accumulatedTimings[5] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 6. rebuild tree
        t0 = std::chrono::high_resolution_clock::now();
        octtree.buildTree(particles);
        accumulatedTimings[6] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 7. mass distribution
        t0 = std::chrono::high_resolution_clock::now();
        octtree.computeMassDistribution(particles);
        accumulatedTimings[7] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 8. reset accelerations
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.ax = p.ay = p.az = 0;
        }
        accumulatedTimings[8] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

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

        for (auto& th : threads)
        {
            th.join();  // sync barrier
        }
        accumulatedTimings[9] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 10. integrate w/ leapfrog (velocity step 2/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogVelStep(TIME_STEP * 0.5f);
        }
        accumulatedTimings[10] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - tpsTimer).count();

        if (elapsed >= 1000 && frameCount > 0)
        {
            double fps = frameCount * 1000.0 / elapsed;
            std::cout << "\nCAM pos: [" << renderer.camera.position.x << ", " << renderer.camera.position.y << ", " << renderer.camera.position.z << "]\n";
            std::cout << "FPS: " << fps << '\n';
            std::cout << "Nodes: " << octtree.nodeCount << "\n";

            double totalAvgTime = 0.0;
            std::array<double, 11> avgTimings = {0.0};

            for (int i = 1; i < 11; ++i) {
                avgTimings[i] = accumulatedTimings[i] / frameCount;
                totalAvgTime += avgTimings[i];
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

            std::cout << "\n===== PROFILING FOR " << particles.size() << " BODIES (AVERAGE PER FRAME) =====\n";

            for (int i = 0; i < 11; ++i)
            {
                double percent = (avgTimings[i] / totalAvgTime) * 100.0;

                std::cout
                    << names[i]
                    << ": "
                    << avgTimings[i]
                    << " ms ("
                    << percent
                    << "%)\n";
            }
            std::cout << "TOTAL AVERAGE frame time: " << totalAvgTime << " ms\n";

            tpsTimer = now;
            frameCount = 0;
            accumulatedTimings.fill(0.0);
        }
    }
    return 0;
}
