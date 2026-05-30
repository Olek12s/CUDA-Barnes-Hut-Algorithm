//
// Created by Oleki on 26.04.2026.
//

#ifndef PARTICLE_H
#define PARTICLE_H

#include <format>

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
        //printf("%.15f\n", az);
        //std::cout << "Acc: " << ax << "\n";
    }

    void leapFrogPosStep(float timeStep) {
        //printf("before: %.15f\n", vx);
        x += vx * timeStep;
        y += vy * timeStep;
        z += vz * timeStep;
       // printf("after: %.15f\n", vx);
    }

    std::string toString() const {
        return std::format(
            "Part{{pos=({:.6f}, {:.6f}, {:.6f}), "
            "vel=({:.11f}, {:.11f}, {:.11f}), "
            "acc=({:.11f}, {:.11f}, {:.11f}), "
            "mass={:.2f}}}",
            x, y, z,
            vx, vy, vz,
            ax, ay, az,
            mass
        );
    }
};


#endif //PARTICLE_H
