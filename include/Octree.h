#ifndef OCTREE_H
#define OCTREE_H
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>
#include "Particle.h"


struct Node {
    Node(int start, int end, int firstChild, float size) : start(start), end(end), firstChild(firstChild), size(size) {}

    int start, end;
    float mass;
    float mcx, mcy, mcz;
    float size;
    // int firstChild;
    // int numChildren;

    int32_t firstChild : 28;
    uint32_t numChildren : 4;

    bool isEmpty() const;
    bool isLeaf();
};

class Octree {
    std::vector<Node> nodes;

public:
    int nodeCount = 0;

    float findRootSize(const std::vector<Particle>& particles);
    void findChildRanges(const std::vector<Particle>& particles, int start, int end, int level, int childStart[8], int childEnd[8]);
    void buildTree(std::vector<Particle> &sortedParticles);
    void computeMassDistribution(const std::vector<Particle>& particles);
    void computeForcesAffectingParticle(int nodeIndex, Particle& particle, const std::vector<Particle>& particles);
};



#endif //OCTREE_H
