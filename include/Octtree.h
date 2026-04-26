#ifndef OCTTREE_H
#define OCTTREE_H
#include <vector>

#include "Particle.h"


struct Node {
    Node(int start, int end, int firstChild, bool isLeaf) : start(start), end(end), firstChild(firstChild), isLeaf(isLeaf) {}


    int start, end;         // start and end index of bodies belonging to the node. Start/end refer to ALREADY SORTED particles (by Morton code)

    float mass;             // current mass of the node
    float mcx, mcy, mcz;   // center of mass position in the node

    float size;                         // size of current node (length of the edge)
    float centerX, centerY, centerZ;    // center of the octant

    unsigned int firstChild;                     // index of first child of current node. Other children's indices are firstchild + n, where n < 8. -1 if child is absent
    bool isLeaf;
};

class Octtree {
    std::vector<Node> nodes;    // whole tree structure sits here with all the informations
    int rootNode = 0;   // root node has always index 0


    // clears nodes vector and reconstructs its content based on given vector in TOP-DOWN range split nature
    void findChildRanges(const std::vector<Particle>& particles,int start, int end,int level,int childStart[8],int childEnd[8]);
    void buildTree(std::vector<Particle> &sortedParticles);


    // void insertBodies();
    // void updateMassDistribution();
    // void updateGravAcceleration();
};



#endif //OCTTREE_H
