#ifndef CUDA_SYSTEM_H
#define CUDA_SYSTEM_H

#include <iostream>
#include <unordered_map>
#define GLM_FORCE_PURE
#include "System.h"

#define CUDA_ENABLE
#ifdef CUDA_ENABLE
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#endif

#define cudaCheck(x) {\
    cudaError_t err = x; \
    if (err != cudaSuccess) {\
    printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0);}}

struct Parameters {
    int particleCount;
    int maxNeighbors;
    float3 gravity;
    float3 bounds_min;
    float3 bounds_max;
    int iterations;
    float dt;
    float h;
    float rest_density;
    float epsilon;
    float k;
    float delta_q;
    float dist_from_bound;
    float viscosity_const;
    float poly6_const;
    float spiky_const;

    int maxGridCount;
    int gridX;
    int gridY;
    int gridZ;
};

class CUDASystem: public System {
public:
    CUDASystem(unsigned numParticles, glm::vec3 bounds_max,
                        std::string config);
    float* getParticlePos();
    unsigned getParticleNum() { return numParticles; }
    void step();
    virtual ~CUDASystem();

private:
    float3* particlePos;
    float3* next_position;
    float3* particleVel;
    float* particleLambda;
    float3* hostParticlePos;
    int* neighborCounts;
    int* neighbors;
    int* gridCount;
    int* grid;
    int gridSize;
};

void update(int gridSize, int particleCount, int iterations, float3 *velocity,
            float3 *position_next, float3 *position, int *neighbor_counts,
            int *neighbors, int *grid_counts, int *grid, float *lambda);

void initialize(struct Parameters *p);


//vector operators
inline __device__ bool operator != (float3 a, float3 b) {
  return !(a.x == b.x && a.y == b.y && a.z == b.z);
}
inline __device__ float operator * (float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ float3 operator * (float a, float3 b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __device__ float3 operator + (float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ float3 operator - (float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __device__ float length2(float3 a) {
  return a * a;
}
inline __device__ float length(float3 a) {
  return sqrt(length2(a));
}
inline __device__ float l2Norm(float3 a, float3 b) {
  return length(a - b);
}
inline __device__ float3 normalize(float3 a) {
  float mag = length(a);
  return make_float3(a.x / mag, a.y / mag, a.z / mag);
}
inline __device__ float3 make_vector(float x) {
  return make_float3(x, x, x);
}
#endif
