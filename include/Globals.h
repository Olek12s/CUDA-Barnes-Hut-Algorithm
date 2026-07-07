#ifndef CONFIG_H
#define CONFIG_H
#include <thread>

inline int SPLIT_AT_LEAF_SIZE = 8;
constexpr int MAX_MORTON_BITS = 21;
constexpr unsigned int MORTON_SCALE = (1u << MAX_MORTON_BITS) - 1u;

inline float EPSILON = 0.35f;
inline float THETA = 0.5f;
inline const float G = 6.674e-11;
inline float EPSILON_SQ = EPSILON * EPSILON;
inline float THETA_SQ = THETA * THETA;

inline float G_MULTIPLIER = 1.0f;
inline float TIME_STEP = 1000.0f;
inline bool ANCHOR = false;
inline float SPREAD_RADIUS = 50.0f;

inline int COM_INTERACTIONS = 0;
inline int DIRECT_INTERACTIONS = 0;

inline const int MAX_HARDWARE_THREADS = std::thread::hardware_concurrency();
inline int NUM_THREADS = MAX_HARDWARE_THREADS;


// ##### VARIABLES FOR IM GUI ##### //
inline float genParticleMass = 1.0f;
inline float genCenterMass = 150000.0f;
inline int genCount = 10000;
inline float minRadius = 1000.0f;
inline float maxRadius = 10000.0f;
inline bool countInteractions = false;
// ##### VARIABLES FOR IM GUI ##### //

#endif //CONFIG_H
