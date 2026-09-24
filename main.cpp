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
#include <fstream>
#include <string>
#include <iomanip>

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

void computeDirectForces(std::vector<Particle>& parts) {
    for (auto& p : parts) {
        p.ax = p.ay = p.az = 0.0f;
    }

#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (intptr_t i = 0; i < static_cast<intptr_t>(parts.size()); i++) {
        for (intptr_t j = 0; j < static_cast<intptr_t>(parts.size()); j++) {
            if (i == j) continue;
            float dx = parts[j].x - parts[i].x;
            float dy = parts[j].y - parts[i].y;
            float dz = parts[j].z - parts[i].z;
            float distSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;
            float invDist = 1.0f / std::sqrt(distSq);
            float invDist3 = invDist * invDist * invDist;
            float factor = G * G_MULTIPLIER * parts[j].mass * invDist3;

            parts[i].ax += dx * factor;
            parts[i].ay += dy * factor;
            parts[i].az += dz * factor;
        }
    }
}

int main() {
    std::vector<Particle> particles;
    Octree octtree;
    Renderer renderer(particles, octtree);
    renderer.init();

    std::array<double, 11> accumulatedTimings = {0.0};
    auto tpsTimer = std::chrono::steady_clock::now();
    int frameCount = 0;

    std::vector<Particle> shadowParticles;
    std::vector<Particle> exactBHState;

    while (!renderer.isTerminated) {
        // 1. render
        auto t0 = std::chrono::high_resolution_clock::now();
        renderer.initFrame();
        renderer.prepareImGuiFrame();
        renderer.renderFrame();         // render
        frameCount++;
        accumulatedTimings[0] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        if (measureAccuracy && !lastMeasureAccuracy) {
            shadowParticles = particles;
            std::fill(errorAccHistory.begin(), errorAccHistory.end(), 0.0f);
            std::fill(errorPosHistory.begin(), errorPosHistory.end(), 0.0f);
            historyOffset = 0;

            std::string filename = csvFileName;
            if (filename.empty()) filename = "pomiary";
            if (filename.find(".csv") == std::string::npos) filename += ".csv";

            csvFile.open(filename, std::ios::out | std::ios::trunc);
            if (csvFile.is_open()) {
                csvFile << std::fixed << std::setprecision(10);
                csvFile << "Tick,Mnoznik_G,Podzial,Theta,Epsilon,Krok_Czasowy,Watki,Liczba_Cial,Blad_E,Blad_Er,Czas_Wykonania\n";
            }

            currentMeasureTick = 0;
            accumErrorAcc = 0.0;
            accumErrorPos = 0.0;

        }
        else if (measureAccuracy && shadowParticles.size() != particles.size()) {
            shadowParticles = particles;
            std::fill(errorAccHistory.begin(), errorAccHistory.end(), 0.0f);
            std::fill(errorPosHistory.begin(), errorPosHistory.end(), 0.0f);
            historyOffset = 0;
        }

        if (!measureAccuracy && lastMeasureAccuracy) {
            if (csvFile.is_open()) {
                csvFile.close();
            }
        }
        lastMeasureAccuracy = measureAccuracy;

        // 2. integrate w/ leapfrog (velocity step 1/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogVelStep(TIME_STEP * 0.5f);
        }
        if (measureAccuracy) for (auto &p : shadowParticles) p.leapFrogVelStep(TIME_STEP * 0.5f);
        accumulatedTimings[1] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 2.5 integrate w/ leapfrog (position step)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogPosStep(TIME_STEP);
        }
        if (measureAccuracy) for (auto &p : shadowParticles) p.leapFrogPosStep(TIME_STEP);
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
        // t0 = std::chrono::high_resolution_clock::now();
        // std::sort(particles.begin(), particles.end(), comp);
        // accumulatedTimings[5] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        t0 = std::chrono::high_resolution_clock::now();

        std::vector<int> perm(particles.size());
        for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int>(i);

        std::sort(perm.begin(), perm.end(), [&](int a, int b) {
            return particles[a].Z_CODE < particles[b].Z_CODE;
        });

        std::vector<Particle> tempParticles;
        tempParticles.reserve(particles.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            tempParticles.push_back(particles[perm[i]]);
        }
        particles = std::move(tempParticles);

        if (measureAccuracy && shadowParticles.size() == particles.size()) {
            std::vector<Particle> tempShadow;
            tempShadow.reserve(shadowParticles.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                tempShadow.push_back(shadowParticles[perm[i]]);
            }
            shadowParticles = std::move(tempShadow);
        }
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

        // // 9. compute forces (multithread)
        // t0 = std::chrono::high_resolution_clock::now();
        // std::vector<std::thread> threads;
        // threads.reserve(NUM_THREADS);
        //
        // auto worker = [&](size_t start, size_t end)
        // {
        //     for (size_t i = start; i < end; i++)
        //     {
        //         octtree.computeForcesAffectingParticle(0, particles[i], particles);
        //     }
        // };
        //
        // size_t n = particles.size();
        // size_t chunk = (n + NUM_THREADS - 1) / NUM_THREADS;
        //
        // for (unsigned int t = 0; t < NUM_THREADS; t++)
        // {
        //     size_t start = t * chunk;
        //     size_t end = std::min(start + chunk, n);
        //     threads.emplace_back(worker, start, end);
        // }
        //
        // for (auto& th : threads)
        // {
        //     th.join();  // sync barrier
        // }
        // accumulatedTimings[9] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        // 9. compute forces
        t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for (intptr_t i = 0; i < static_cast<intptr_t>(particles.size()); ++i) {
            octtree.computeForcesAffectingParticle(0, particles[i], particles);
        }
        accumulatedTimings[9] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();


        if (measureAccuracy && !particles.empty()) {
            computeDirectForces(shadowParticles);

            exactBHState = particles;
            computeDirectForces(exactBHState);

            float totalE = 0.0f;
            float totalEr = 0.0f;
            size_t n = particles.size();

            for (size_t i = 0; i < n; i++) {
                float diffX = particles[i].ax - exactBHState[i].ax;
                float diffY = particles[i].ay - exactBHState[i].ay;
                float diffZ = particles[i].az - exactBHState[i].az;
                float diffMag = std::sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);

                float dirMag = std::sqrt(exactBHState[i].ax*exactBHState[i].ax +
                                         exactBHState[i].ay*exactBHState[i].ay +
                                         exactBHState[i].az*exactBHState[i].az);

                if (dirMag > 1e-10f) {
                    totalE += (diffMag / dirMag);
                }

                float rx = particles[i].x - shadowParticles[i].x;
                float ry = particles[i].y - shadowParticles[i].y;
                float rz = particles[i].z - shadowParticles[i].z;
                totalEr += std::sqrt(rx*rx + ry*ry + rz*rz);
            }

            currentErrorAcc = totalE / n;
            currentErrorPos = totalEr / n;

            accumErrorAcc += currentErrorAcc;
            accumErrorPos += currentErrorPos;

            errorAccHistory[historyOffset] = currentErrorAcc;
            errorPosHistory[historyOffset] = currentErrorPos;
            historyOffset = (historyOffset + 1) % errorAccHistory.size();
        }



        // 10. integrate w/ leapfrog (velocity step 2/2)
        t0 = std::chrono::high_resolution_clock::now();
        for (auto &p : particles) {
            p.leapFrogVelStep(TIME_STEP * 0.5f);
        }
        if (measureAccuracy) for (auto &p : shadowParticles) p.leapFrogVelStep(TIME_STEP * 0.5f);
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

            if (measureAccuracy && csvFile.is_open()) {
                double avgE = accumErrorAcc / frameCount;
                double avgEr = accumErrorPos / frameCount;

                csvFile << currentMeasureTick << ","
                        << G_MULTIPLIER << ","
                        << SPLIT_AT_LEAF_SIZE << ","
                        << THETA << ","
                        << EPSILON << ","
                        << TIME_STEP << ","
                        << NUM_THREADS << ","
                        << particles.size() << ","
                        << avgE << ","
                        << avgEr << ","
                        << totalAvgTime << "\n";
                csvFile.flush();

                currentMeasureTick++;
            }

            accumErrorAcc = 0.0;
            accumErrorPos = 0.0;

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
