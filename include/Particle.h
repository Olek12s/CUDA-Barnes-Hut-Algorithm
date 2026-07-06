//
// Created by Oleki on 26.04.2026.
//

#ifndef PARTICLE_H
#define PARTICLE_H

#include <format>

struct Particle {
    float x, y, z;              // position
    float vx, vy, vz;           // velocity
    float ax, ay, az;           // acceleration
    float mass;                 // mass
    uint64_t Z_CODE : 63;       // morton code
    uint64_t anchored : 1;      // anchor - takes Most Significant Bit

    Particle(float x, float y, float z, float m, float vx, float vy, float vz):
    x(x),y(y),z(z), vx(vx), vy(vy), vz(vz), ax(0), ay(0), az(0), mass(m), Z_CODE(0), anchored(false) {}


    void setAnchored(bool anchor) {
        anchored = anchor;
    }

    bool isAnchored() const {
        return anchored;
    }

    void leapFrogVelStep(float halfTimeStep) {
        if (isAnchored()) return;

        vx += ax * halfTimeStep;
        vy += ay * halfTimeStep;
        vz += az * halfTimeStep;
    }

    void leapFrogPosStep(float timeStep) {
        if (isAnchored()) return;

        x += vx * timeStep;
        y += vy * timeStep;
        z += vz * timeStep;
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
