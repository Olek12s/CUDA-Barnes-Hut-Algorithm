#include "Octtree.h"

#include <stack>

#include "Config.h"

bool Node::isEmppty() const {
    return start == -1;
}


// result example for node [0, 16):
// childStart = [0, 4, 4, 7, 10, 10, 12, 14]
// childEnd   = [4, 4, 7, 10, 10, 12, 14, 16]
void Octtree::findChildRanges(const std::vector<Particle> &particles, int start, int end, int level, int *childStart, int *childEnd) {
    for (int i = 0; i < 8; i++) {
        childStart[i] = -1;
        childEnd[i] = -1;
    }

    int shift = 3 * (21 - level - 1);   // MAX_MORTON_BITS = 21

    for (int i = start; i < end; i++) {
        unsigned int octant = (particles[i].Z_CODE >> shift) & 7;   // 7 == 0b111

        if (childStart[octant] == -1) {
            childStart[octant] = i;
        }
        childEnd[octant] = i + 1;
    }
}


void Octtree::buildTree(std::vector<Particle> &sortedParticles) {
    nodes.clear();  // clear off nodes vector

    Node root(0, sortedParticles.size(), -1, true);
    nodes.push_back(root);

    std::stack<std::pair<int, int>> stack; // first - nodeIndex, second - level // stack of nodes
    stack.push({0, 0});


    while (!stack.empty()) {
        // auto [nodeIndex, level] = stack.top();
        int nodeIndex = stack.top().first;
        int level = stack.top().second;
        stack.pop();

        Node &node = nodes[nodeIndex];
        int count = node.end - node.start;

        if (count <= SPLIT_AT_LEAF_SIZE || level >= MAX_MORTON_BITS) {
            if (level >= MAX_MORTON_BITS) std::cout << "Level over " << MAX_MORTON_BITS << "." << " \n";
            node.isLeaf = true;
            node.firstChild = -1;
            continue;
        }
        std::cout << "Exceeded MAX_LEAF_SIZE. Dividing node into 8 children \n";

        int childStart[8];
        int childEnd[8];

        findChildRanges(sortedParticles, node.start, node.end, level, childStart, childEnd);

        node.firstChild = nodes.size();
        node.isLeaf = false;


        // Create up to 8 child nodes representing spatial octants
        // Some children may be empty if no particles fall into that region

        node.firstChild = nodes.size();
        node.isLeaf = false;

        for (int i = 0; i < 8; i++)
        {
            if (childStart[i] == -1)
            {
                nodes.push_back(Node(-1, -1, -1, true)); // empty leaf
            }
            else
            {
                nodes.push_back(Node(childStart[i], childEnd[i], -1, false));   // leaf with data (no division yet thus firstChild -1)
            }

            stack.push({node.firstChild + i, level + 1});   // push processed node on stack
        }
    }
}

void Octtree::computeMassDistribution(const std::vector<Particle> &particles) {
        if (nodes.empty()) {
            std::cout << "Nodes are empty.\n";
            return;
        }

    for (int i = nodes.size() - 1; i >= 0; i--) {
        Node &node = nodes[i];

        node.mass = 0;
        node.mcx = node.mcy = node.mcz = 0;

        // if leaf - calculate node's mass and COM based on particles inside that node
        if (node.isLeaf) {
            if (node.isEmppty()) {                  // TODO: test it by commenting out
                continue;
            }

            for (int p = node.start; p < node.end; p++) {
                const Particle &particle = particles[p];

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

            for (int j = 0; j < 8; j++)
            {
                Node &child = nodes[first + j];

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