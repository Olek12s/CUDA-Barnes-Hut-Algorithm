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

void Octtree::computeMassDistribution() {
        if (nodes.empty()) {
            std::cout << "Nodes are empty.\n";
            return;
        }

}