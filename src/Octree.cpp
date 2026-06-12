#include "Octree.h"

#include <stack>
#include <iostream>
#include <array>
#include <algorithm>

#include "Globals.h"


bool Node::isEmpty() const {
    return start == -1;
}

bool Node::isLeaf() {
    return firstChild == -1;
}

float Octree::findRootSize(const std::vector<Particle>& particles) {
    std::array<std::pair<float, float>, 3> bounds =
   {{
       {std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest()},

       {std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest()},

       {std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest()}
   }};

    for (auto &p : particles) {
        // x
        bounds[0].first = std::min(bounds[0].first, p.x);
        bounds[0].second = std::max(bounds[0].second, p.x);

        // y
        bounds[1].first = std::min(bounds[1].first, p.y);
        bounds[1].second = std::max(bounds[1].second, p.y);

        // z
        bounds[2].first = std::min(bounds[2].first, p.z);
        bounds[2].second = std::max(bounds[2].second, p.z);
    }

    float dx = bounds[0].second - bounds[0].first;
    float dy = bounds[1].second - bounds[1].first;
    float dz = bounds[2].second - bounds[2].first;

    float rootSize = std::max({dx, dy, dz});

    if (rootSize == 0.0f) {
        rootSize = 1.0f;    // edge case where all elements have same position
    }
    return rootSize;
}

// result example for node [0, 16):
// childStart = [0, 4, 4, 7, 10, 10, 12, 14]
// childEnd   = [4, 4, 7, 10, 10, 12, 14, 16]
void Octree::findChildRanges(const std::vector<Particle> &particles, int start, int end, int level, int *childStart, int *childEnd) {
    for (int i = 0; i < 8; i++) {
        childStart[i] = -1;
        childEnd[i] = -1;
    }

    int shift = 3 * (21 - level - 1); // MAX_MORTON_BITS = 21. shift is used to check only bit at the given level

    for (int i = start; i < end; i++) {
        unsigned int octant = (particles[i].Z_CODE >> shift) & 7;   // 7 == 0b111

        if (childStart[octant] == -1) {
            childStart[octant] = i;
        }
        childEnd[octant] = i + 1;
    }
}


void Octree::buildTree(std::vector<Particle>& sortedParticles) {
    nodes.clear();  // clear off nodes vector
    nodeCount = 0;
    COM_INTERACTIONS = 0;
    DIRECT_INTERACTIONS = 0;
    float rootSize = findRootSize(sortedParticles);  // find rootSize

    Node root(0, sortedParticles.size(), -1, rootSize);
    nodes.push_back(root);
    nodeCount++;

    std::stack<std::pair<int, int>> stack; // first - nodeIndex, second - level // stack of nodes
    stack.push({0, 0});

    while (!stack.empty()) {

        int nodeIndex = stack.top().first;
        int level = stack.top().second;
        stack.pop();

        int count = nodes[nodeIndex].end - nodes[nodeIndex].start;

        if (count <= SPLIT_AT_LEAF_SIZE || level >= MAX_MORTON_BITS) {
            //if (level >= MAX_MORTON_BITS) std::cout << "Level over " << MAX_MORTON_BITS << "." << " \n";
            nodes[nodeIndex].firstChild = -1;
            continue;
        }
        //std::cout << "Exceeded MAX_LEAF_SIZE. Dividing node into 8 children \n";

        int childStart[8];
        int childEnd[8];

        findChildRanges(sortedParticles, nodes[nodeIndex].start, nodes[nodeIndex].end, level, childStart, childEnd);
        nodes[nodeIndex].firstChild = nodes.size();

        float childSize = nodes[nodeIndex].size * 0.5f;
        int childCount = 0;

        // Nodes are not created if they would carry no particles
        for (int i = 0; i < 8; i++)
        {
            if (childStart[i] != -1)    // ignore empty octants - don't create empty octants with no particles inside
            {
                nodes.push_back(Node(childStart[i], childEnd[i], -1, childSize)); // empty leaf
                nodeCount++;

                stack.push({nodes[nodeIndex].firstChild + childCount, level + 1});
                childCount++;
            }
            nodes[nodeIndex].numChildren = childCount;
        }
    }
}

void Octree::computeMassDistribution(const std::vector<Particle> &particles) {
    for (int i = nodes.size() - 1; i >= 0; i--) {
        Node& node = nodes[i];
        node.mass = 0;
        node.mcx = node.mcy = node.mcz = 0;

        // if leaf - calculate node's mass and COM based on particles inside that node
        if (node.isLeaf()) {
            if (node.isEmpty()) continue;

            for (int p = node.start; p < node.end; p++) {
                if (node.start == -1 || node.end == -1) continue;   // if nmode is empty
                const Particle& particle = particles[p];

                // update COM
                node.mass += particle.mass;
                node.mcx += particle.x * particle.mass;
                node.mcy += particle.y * particle.mass;
                node.mcz += particle.z * particle.mass;
            }
            if (node.mass > 0)
            {
                node.mcx /= node.mass;
                node.mcy /= node.mass;
                node.mcz /= node.mass;
            }
        }
        else    // else calculate first node's children and accumulate COMs
        {
            float totalMass = 0;
            float cx = 0, cy = 0, cz = 0;

            int first = node.firstChild;

            for (int j = 0; j < node.numChildren; j++)
            {
                Node& child = nodes[first + j];

                totalMass += child.mass;
                cx += child.mcx * child.mass;
                cy += child.mcy * child.mass;
                cz += child.mcz * child.mass;
            }
            node.mass = totalMass;

            if (totalMass > 0)
            {
                node.mcx = cx / totalMass;
                node.mcy = cy / totalMass;
                node.mcz = cz / totalMass;
            }
        }
    }
}

void Octree::computeForcesAffectingParticle(int nodeIndex, Particle &particle, const std::vector<Particle> &particles) {
    Node& node = nodes[nodeIndex];

    if (node.mass == 0) {
        return;
    }

    // vector from particle position to node center of mass
    float dx = node.mcx - particle.x;
    float dy = node.mcy - particle.y;
    float dz = node.mcz - particle.z;

    // ignore A <-> A and leaf nodes
    if (node.isLeaf() && node.end - node.start == 1 && &particles[node.start] == &particle)
    {
        return;
    }

    float distSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;
    float sizeSq = node.size * node.size;

    if (node.isLeaf() || (sizeSq < distSq * THETA_SQ)) {
        float invDist = 1.0f / sqrtf(distSq);
        float invDist3 = invDist * invDist * invDist;
        float factor = G * G_MULTIPLIER * node.mass * invDist3; // check the node's COM

        if (countInteractions) {
            if (node.end - node.start > 1) {
                COM_INTERACTIONS++;
            } else {
                DIRECT_INTERACTIONS++;
            }
        }

        particle.ax += dx * factor;
        particle.ay += dy * factor;
        particle.az += dz * factor;
    }
    else {  // traverse children TOP-BOTTOM
        if (node.firstChild == -1) return;

        for (int i = 0; i < node.numChildren; i++)
        {
            computeForcesAffectingParticle(node.firstChild + i, particle, particles);
        }
    }
}
