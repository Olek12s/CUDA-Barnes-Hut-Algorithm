#ifndef CONFIG_H
#define CONFIG_H

static constexpr int SPLIT_AT_LEAF_SIZE = 1024;
constexpr int MAX_MORTON_BITS = 21;
float EPSILON = 0.005;   // smaller - sharper physics | larger - smoother and more stable physics | Prevents infinite forces when distance r ~0. Prefferable [0.01-0.5]
float THETA = 0.5;      // smaller - more precise, but slower | larger - less precise, but faster | prefferable - [0.3 0.7]
float G = 6.674e-11;    // real constant is 6.674e-11
float EPSILON_SQ = EPSILON * EPSILON;




#endif //CONFIG_H
