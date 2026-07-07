#include "Octree.h"

#include <stack>
#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>

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
        rootSize = 1.0f;
    }
    return rootSize;
}

void Octree::findChildRanges(const std::vector<Particle> &particles, int start, int end, int level, int *childStart, int *childEnd) {
    for (int i = 0; i < 8; i++) {
        childStart[i] = -1;
        childEnd[i] = -1;
    }

    int shift = 3 * (21 - level - 1);

    for (int i = start; i < end; i++) {
        unsigned int octant = (particles[i].Z_CODE >> shift) & 7;   // 7 == 0b111

        if (childStart[octant] == -1) {
            childStart[octant] = i;
        }
        childEnd[octant] = i + 1;
    }
}


void Octree::buildTree(std::vector<Particle>& sortedParticles) {
    nodes.clear();
    nodeCount = 0;
    COM_INTERACTIONS = 0;
    DIRECT_INTERACTIONS = 0;
    float rootSize = findRootSize(sortedParticles);

    Node root(0, sortedParticles.size(), -1, rootSize);
    nodes.push_back(root);
    nodeCount++;

    std::stack<std::pair<int, int>> stack;
    stack.push({0, 0});

    while (!stack.empty()) {

        int nodeIndex = stack.top().first;
        int level = stack.top().second;
        stack.pop();

        int count = nodes[nodeIndex].end - nodes[nodeIndex].start;

        if (count <= SPLIT_AT_LEAF_SIZE || level >= MAX_MORTON_BITS) {
            nodes[nodeIndex].firstChild = -1;
            continue;
        }

        int childStart[8];
        int childEnd[8];

        findChildRanges(sortedParticles, nodes[nodeIndex].start, nodes[nodeIndex].end, level, childStart, childEnd);
        nodes[nodeIndex].firstChild = nodes.size();

        float childSize = nodes[nodeIndex].size * 0.5f;
        int childCount = 0;

        for (int i = 0; i < 8; i++)
        {
            if (childStart[i] != -1)
            {
                nodes.push_back(Node(childStart[i], childEnd[i], -1, childSize));
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

        if (node.isLeaf()) {
            if (node.isEmpty()) continue;

            for (int p = node.start; p < node.end; p++) {
                if (node.start == -1 || node.end == -1) continue;
                const Particle& particle = particles[p];

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
        else
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

    float dx = node.mcx - particle.x;
    float dy = node.mcy - particle.y;
    float dz = node.mcz - particle.z;

    float distSq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;
    float sizeSq = node.size * node.size;

    if (sizeSq < distSq * THETA_SQ) {
        float invDist = 1.0f / sqrtf(distSq);
        float invDist3 = invDist * invDist * invDist;
        float factor = G * G_MULTIPLIER * node.mass * invDist3;

        particle.ax += dx * factor;
        particle.ay += dy * factor;
        particle.az += dz * factor;

        if (countInteractions) COM_INTERACTIONS++;
    }
    else {
        if (node.isLeaf()) {
            for (int p = node.start; p < node.end; p++) {
                if (p == -1) continue;
                const Particle& target = particles[p];

                if (&target == &particle) continue;

                float pdx = target.x - particle.x;
                float pdy = target.y - particle.y;
                float pdz = target.z - particle.z;

                float pDistSq = pdx*pdx + pdy*pdy + pdz*pdz + EPSILON_SQ;
                float pInvDist = 1.0f / sqrtf(pDistSq);
                float pInvDist3 = pInvDist * pInvDist * pInvDist;
                float pFactor = G * G_MULTIPLIER * target.mass * pInvDist3;

                particle.ax += pdx * pFactor;
                particle.ay += pdy * pFactor;
                particle.az += pdz * pFactor;

                if (countInteractions) DIRECT_INTERACTIONS++;
            }
        }
        else {
            for (int i = 0; i < node.numChildren; i++) {
                computeForcesAffectingParticle(node.firstChild + i, particle, particles);
            }
        }
    }
}
