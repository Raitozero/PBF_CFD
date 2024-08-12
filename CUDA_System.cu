#include "CUDA_System.h"
#include <math.h>
#include <random>

CUDASystem::CUDASystem(unsigned numParticles, glm::vec3 bounds_max, std::string config):System(numParticles, bounds_max) {
  //Set the parameters
  Parameters par;
  par.particleCount = numParticles;
  par.maxNeighbors = maxNeighbors;
  par.gravity = make_float3(gravity.x, gravity.y, gravity.z);
  par.bounds_min = make_float3(bounds_min.x, bounds_min.y, bounds_min.z);
  par.bounds_max = make_float3(bounds_max.x, bounds_max.y, bounds_max.z);
  par.iterations = iterations;
  par.dt = dt;
  par.h = h;
  par.rest_density = rest_density;
  par.epsilon = epsilon;
  par.k = k;
  par.dist_from_bound = dist_from_bound;
  par.delta_q = delta_q;
  par.viscosity_const = viscosity_const;
  par.poly6_const = poly6_const;
  par.spiky_const = spiky_const;

  //Set the grid
  par.maxGridCount = maxNeighbors;
  par.gridX = int(ceil((bounds_max.x - bounds_min.x) / h));
  par.gridY = int(ceil((bounds_max.y - bounds_min.y) / h));
  par.gridZ = int(ceil((bounds_max.z - bounds_min.z) / h));

  hostParticlePos = (float3 *)malloc(numParticles * sizeof(float3));
  std::default_random_engine generator;

  if (config == "dam") {
    std::uniform_real_distribution<float> distributionX(bounds_min.x + 0.1, bounds_min.x + 40);
    std::uniform_real_distribution<float> distributionY(bounds_min.y + 0.1, bounds_max.y - 0.1);
    std::uniform_real_distribution<float> distributionZ(bounds_min.z + 0.1, bounds_max.z - 0.1);
    for (int i = 0; i < numParticles; i++) {
      hostParticlePos[i] = make_float3(distributionX(generator), distributionY(generator), distributionZ(generator));
    }
  } else if (config == "sphere") {
    float r = std::min(std::min(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y), bounds_max.z - bounds_min.z) / 2.0;
    float3 offset = make_float3((bounds_max.x - bounds_min.x) / 2.0, (bounds_max.y - bounds_min.y) / 2.0, (bounds_max.z - bounds_min.z) / 2.0);
    std::uniform_real_distribution<float> distributionR(-r, r);
    float x, y, z;
    for (int i = 0; i < numParticles; i++) {
      do {
        x = distributionR(generator);
        y = distributionR(generator);
        z = distributionR(generator);
      } while (x * x + y * y + z * z >= r * r);
      hostParticlePos[i] = make_float3(x, y, z) + offset;
    }
  } else {
    std::uniform_real_distribution<float> distribution(bounds_min.x + 5, bounds_max.x - 5);
    for (int i = 0; i < numParticles; i++) {
      hostParticlePos[i] =
          make_float3(distribution(generator), distribution(generator),
                      distribution(generator));
    }
  }

  gridSize = par.gridX * par.gridY * par.gridZ;

  cudaCheck(cudaMalloc((void **)&particlePos, numParticles * sizeof(float3)));
  cudaCheck(cudaMalloc((void **)&particleVel, numParticles * sizeof(float3)));
  cudaCheck(cudaMalloc((void **)&next_position, numParticles * sizeof(float3)));
  cudaCheck(cudaMalloc((void **)&particleLambda, numParticles * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&neighborCounts, numParticles * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&neighbors, numParticles * maxNeighbors * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&gridCount, gridSize * sizeof(int)));
  cudaCheck(cudaMalloc((void **)&grid, gridSize * par.maxGridCount * sizeof(int)));

  cudaCheck(cudaMemset(particlePos, 0, numParticles * sizeof(float3)));
  cudaCheck(cudaMemset(particleVel, 0, numParticles * sizeof(float3)));
  cudaCheck(cudaMemset(next_position, 0, numParticles * sizeof(float3)));
  cudaCheck(cudaMemset(particleLambda, 0, numParticles * sizeof(float)));
  cudaCheck(cudaMemset(neighborCounts, 0, numParticles * sizeof(int)));
  cudaCheck(cudaMemset(neighbors, 0, numParticles * maxNeighbors * sizeof(int)));
  cudaCheck(cudaMemset(gridCount, 0, gridSize * sizeof(int)));
  cudaCheck(cudaMemset(grid, 0, gridSize * par.maxGridCount * sizeof(int)));

  cudaCheck(cudaMemcpy(particlePos, hostParticlePos, numParticles * sizeof(float3), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(next_position, hostParticlePos, numParticles * sizeof(float3), cudaMemcpyHostToDevice));
  initialize(&par);
}

CUDASystem::~CUDASystem() {
  cudaCheck(cudaFree(particlePos));
  cudaCheck(cudaFree(particleVel));
  cudaCheck(cudaFree(next_position));
  cudaCheck(cudaFree(particleLambda));
  cudaCheck(cudaFree(neighborCounts));
  cudaCheck(cudaFree(neighbors));
  cudaCheck(cudaFree(gridCount));
  cudaCheck(cudaFree(grid));
  free(hostParticlePos);
}

float *CUDASystem::getParticlePos() {
#ifdef DEVICE_RENDER
  return (float *)particlePos;

#else
  cudaCheck(cudaMemcpy(hostParticlePos, particlePos,
                       numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
  return &hostParticlePos[0].x;

#endif
}

void CUDASystem::step() {
  update(gridSize, numParticles, iterations, particleVel, next_position,
         particlePos, neighborCounts, neighbors, gridCount, grid,
         particleLambda);
}
