//
// Created by Oleki on 26.04.2026.
//

#ifndef PARTICLE_H
#define PARTICLE_H

struct Particle {
    float x, y, z;      // position
    float vx, vy, vz;   // velocity
    float ax, ay, az;   // acceleration
    float mass;
    uint64_t Z_CODE;
};


#endif //PARTICLE_H
