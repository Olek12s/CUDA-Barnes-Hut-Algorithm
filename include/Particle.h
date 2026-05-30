//
// Created by Oleki on 26.04.2026.
//

#ifndef PARTICLE_H
#define PARTICLE_H

struct Particle {
    float x, y, z;      // position
    float vx, vy, vz;   // velocity
    float ax, ay, az;   // acceleration
    float mass;         // mass
    uint64_t Z_CODE;    // morton code


    void euler(float timeStep) {
        vx += ax * timeStep;
        vy += ay * timeStep;
        vz += az * timeStep;

        x += vx * timeStep;
        y += vy * timeStep;
        z += vz * timeStep;
    }

    void leapFrogVelStep(float halfTimeStep) {
        vx += ax * halfTimeStep;
        vy += ay * halfTimeStep;
        vz += az * halfTimeStep;
    }

    void leapFrogPosStep(float timeStep) {
        x += vx * timeStep;
        y += vy * timeStep;
        z += vz * timeStep;
    }
};


#endif //PARTICLE_H
