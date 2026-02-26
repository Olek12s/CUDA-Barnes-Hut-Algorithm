#ifndef OCTTREE_H
#define OCTTREE_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct Body {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 acceleration;
    glm::vec3 lastAcceleration; // optional?
    float mass;
};

struct Node {
    int children[8];

    const int* getChildren() const { return children; }
};

class Octtree {
    int rootNode = 0;   // root node has always index 0



    void insertBodies();
    void updateMassDistribution();
    void updateGravAcceleration();
};



#endif //OCTTREE_H
