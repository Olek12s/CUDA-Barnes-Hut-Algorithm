#ifndef OCTTREE_H
#define OCTTREE_H
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>
#include "Particle.h"


struct Node {
    Node(int start, int end, int firstChild, float size, bool isLeaf) : start(start), end(end), firstChild(firstChild), size(size), isLeaf(isLeaf) {}

    int start, end;         // start and end index of bodies belonging to the node. Start/end refer to ALREADY SORTED particles (by Morton code)

    float mass;             // current mass of the node
    float mcx, mcy, mcz;   // center of mass position in the node

    float size;                         // size of current node (length of the edge)
    float centerX, centerY, centerZ;    // center of the octant

    int firstChild;                     // index of first child of current node. Other children's indices are firstchild + n, where n < 8. -1 if child is absent
    bool isLeaf;

    bool isEmpty() const;
};

class Octtree {
    std::vector<Node> nodes;    // whole tree structure sits here with all the informations
    int rootNode = 0;           // root node has always index 0

public:

    float findRootSize(const std::vector<Particle>& particles);


    // For the currently processed node (ONLY ONE NODE) range [start, end)
    // finds index ranges belonging to its 8 children at Vec<Particles> vector.
    //
    // particles MUST already be sorted by Morton code.
    //
    // Result:
    // childStart[i] -> first index of child i
    // childEnd[i]   -> index AFTER the last element of child i
    //
    // If a child does not exist:
    // childStart[i] == -1
    // childEnd[i]   == -1
    // example:
    // childStart = [0, 2, -1, 5, -1, -1, 7, 9]
    // childEnd = [2, 5, -1, 7, -1, -1, 9,10]
    void findChildRanges(const std::vector<Particle>& particles, int start, int end, int level, int childStart[8], int childEnd[8]);

    // clears nodes vector and reconstructs its content based on given vector in TOP-DOWN range split nature
    void buildTree(std::vector<Particle> &sortedParticles);

    // computes mass dist. among tree nodes in BOTTOM-UP nature. If node leaf - bodies of this node are taking in calculation, else calculate node's children first
    void computeMassDistribution(const std::vector<Particle>& particles);

    void computeForcesAffectingParticle(int nodeIndex, Particle& particle, float& ax, float& ay, float& az, const std::vector<Particle>& particles);

    // void insertBodies();
    // void updateMassDistribution();
    // void updateGravAcceleration();
};



#endif //OCTTREE_H
