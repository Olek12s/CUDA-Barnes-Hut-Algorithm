#ifndef CONFIG_H
#define CONFIG_H
#include <thread>

static int SPLIT_AT_LEAF_SIZE = 2048;
constexpr int MAX_MORTON_BITS = 21; // Z_CODE has 64 unsigned bit type - code is defined by 3 values, thus maximum morton bits is 64/3 = 21
constexpr unsigned int MORTON_SCALE = (1u << MAX_MORTON_BITS) - 1u; // 2097151, or std::pow(2, 21 //Morton_SCALE is in other words the biggest digit possible to encode on 21 btis

inline float EPSILON = 0.35f;        // smaller - sharper physics | larger - smoother and more stable physics | Prevents infinite forces when distance r ~0. Prefferable [0.01-0.5]
inline float THETA = 0.5f;           // smaller - more precise, but slower | larger - less precise, but faster | prefferable - [0.3 0.7]
inline const float G = 6.674e-10;    // real constant is 6.674e-11
inline float EPSILON_SQ = EPSILON * EPSILON;
inline float THETA_SQ = THETA * THETA;

inline float G_MULTIPLIER = 1.0f;
inline float TIME_STEP = 1000.0f;

inline const int MAX_HARDWARE_THREADS = std::thread::hardware_concurrency();
inline int NUM_THREADS = MAX_HARDWARE_THREADS / 2;

#endif //CONFIG_H
