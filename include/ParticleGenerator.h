//
// Created by Oleki on 10.06.2026.
//

#ifndef PARTICLEGENERATOR_H
#define PARTICLEGENERATOR_H
#include "Particle.h"
#include <vector>


class ParticleGenerator {
public:
    static void addParticle(std::vector<Particle>& particles, float x, float y, float z, float mass, float vx = 0.0f, float vy = 0.0f, float vz = 0.0f);
    static void createFlatRectangle(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float vx = 0.0f, float vy = 0.0f, float vz = 0.0f);
    static void createCube(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float vx = 0.0f, float vy = 0.0f, float vz = 0.0f);

    static void createDisc(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float centerMass, float minR, float maxR, float vx, float vy, float vz);
    static void createSphere(std::vector<Particle>& particles, float x, float y, float z, int count, float particleMass, float centerMass, float minR, float maxR, float vx, float vy, float vz);

};



#endif //PARTICLEGENERATOR_H
