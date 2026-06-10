//
// Created by Oleki on 10.06.2026.
//

#include "ParticleGenerator.h"
#include <random>
#include <cmath>

#include "Config.h"

constexpr float SPREAD_RADIUS = 50.0f;

void ParticleGenerator::addParticle(std::vector<Particle>& particles, float x, float y, float z, float mass, float vx, float vy, float vz) {
    particles.push_back(Particle(x, y, z, mass, vx, vy, vz));
}

void ParticleGenerator::createFlatRectangle(std::vector<Particle> &particles, float x, float y, float z, int count, float particleMass, float vx, float vy, float vz) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-SPREAD_RADIUS, SPREAD_RADIUS);

    for (int i = 0; i < count; i++) {
        particles.push_back(Particle(x + dist(gen), y + dist(gen), z, particleMass, vx, vy, vz));
    }
}

void ParticleGenerator::createCube(std::vector<Particle> &particles, float x, float y, float z, int count, float particleMass, float vx, float vy, float vz) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-SPREAD_RADIUS, SPREAD_RADIUS);

    for (int i = 0; i < count; i++) {
        particles.push_back(Particle(x + dist(gen), y + dist(gen), z + dist(gen), particleMass, vx, vy, vz));
    }
}

void ParticleGenerator::createDisc(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float centerMass, float minR, float maxR, float vx, float vy, float vz) {
    Particle center(x, y, z, centerMass, vx, vy, vz);
    if(ANCHOR) center.setAnchored(true);
    particles.push_back(center); // center


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

    for (int i = 1; i < count; i++) {
        float angle = angleDist(gen);
        float r = minR + (maxR - minR) * std::sqrt(uDist(gen)); // random radius between minR maxR

        float orbitalSpeed = std::sqrt(G * G_MULTIPLIER * centerMass / (r + 0.1f));

        // Velocity vector perpendicular to the radius in XY surface
        float vpx = -sin(angle) * orbitalSpeed*4 + vx;
        float vpy =  cos(angle) * orbitalSpeed*4 + vy;

        particles.push_back(Particle(x + r * cos(angle), y + r * sin(angle), z, particleMass, vpx, vpy, vz));
    }
}

void ParticleGenerator::createSphere(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float centerMass, float minR, float maxR, float vx, float vy, float vz) {
    Particle center(x, y, z, centerMass, vx, vy, vz);
    if(ANCHOR) center.setAnchored(true);
    particles.push_back(center); // center

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> phiDist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> costhetaDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

    for (int i = 1; i < count; i++) {
        float phi = phiDist(gen);
        float costheta = costhetaDist(gen);
        float theta = acos(costheta);
        float r = minR + (maxR - minR) * std::cbrt(uDist(gen)); // cubic root

        float px = x + r * sin(theta) * cos(phi);
        float py = y + r * sin(theta) * sin(phi);
        float pz = z + r * costheta;

        particles.push_back(Particle(px, py, pz, particleMass, vx, vy, vz));
    }
}




