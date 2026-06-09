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

    // Particle() {};
    // Particle(float x, float y, float z): x(x),y(y),z(z) {}

    Particle(): x(0), y(0), z(0),vx(0), vy(0), vz(0),ax(0), ay(0), az(0),mass(0),Z_CODE(0) {}
    Particle(float x, float y, float z): x(x),y(y),z(z),  vx(0), vy(0), vz(0),ax(0), ay(0), az(0),mass(1.0f), Z_CODE(0) {}
    Particle(float x, float y, float z, float m): x(x),y(y),z(z),  vx(0), vy(0), vz(0),ax(0), ay(0), az(0),mass(m), Z_CODE(0) {}
    Particle(float x, float y, float z, float m, float vx, float vy, float vz): x(x),y(y),z(z),  vx(vx), vy(vy), vz(vz),ax(0), ay(0), az(0),mass(m), Z_CODE(0) {}

    void euler(float timeStep) {
        vx += ax * timeStep;
        vy += ay * timeStep;
        vz += az * timeStep;

        x += vx * timeStep;
        y += vy * timeStep;
        z += vz * timeStep;
    }

    void leapFrogVelStep(float halfTimeStep) {
        // float max = 1000000;

        vx += ax * halfTimeStep;
        vy += ay * halfTimeStep;
        vz += az * halfTimeStep;

        // if (vx > max) vx = max;
        // if (vy > max) vy = max;
        // if (vz > max) vz = max;

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
            "acc=({:.32f}, {:.32f}, {:.32f}), "
            "mass={:.2f}}}",
            x, y, z,
            vx, vy, vz,
            ax, ay, az,
            mass
        );
    }
};


#endif //PARTICLE_H
