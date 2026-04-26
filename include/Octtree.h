#ifndef OCTTREE_H
#define OCTTREE_H
#include <vector>


struct Node {
    int start, end;         // start and end index of bodies belonging to the node. Start/end refer to ALREADY SORTED particles (by Morton code)

    float mass;             // current mass of the node
    float mcx, mcy, mcz;   // center of mass position in the node

    float size;                         // size of current node (length of the edge)
    float centerX, centerY, centerZ;    // center of the octant
};

class Octtree {
    std::vector<Node> nodes;    // whole tree structure sits here with all the informations
    int rootNode = 0;   // root node has always index 0



    void buildTree(std::vector<Particle>& sortedParticles)
    // void insertBodies();
    // void updateMassDistribution();
    // void updateGravAcceleration();
};



#endif //OCTTREE_H
