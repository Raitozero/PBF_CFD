#include "CUDA_System.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <device_launch_parameters.h>

#define EPSILON 0.0000000001f
#define NUM_THREADS 256
#define APPLY_FORCES_THREADS 1024
#define OFFSET_KERNEL_THREADS 1024

__constant__ struct Parameters par;

__device__ float poly6(float3 r) {
    float norm_coeff = par.h * par.h - r * r;
    if (norm_coeff <= 0) return 0.0f;
    if (r.x == 0.0f && r.y == 0.0f && r.z == 0.0f) return 0.0f;
    return par.poly6_const * norm_coeff * norm_coeff * norm_coeff;
}

__device__ float3 spiky_prime(float3 r) {
    float3 r_norm = normalize(r);
    float norm_coeff = par.h - length(r);
    if (norm_coeff <= 0) return make_vector(0.0f);
    if (r.x == 0.0f && r.y== 0.0f && r.z == 0.0f) return make_vector(0.0f);
    return par.spiky_const * norm_coeff * norm_coeff * r_norm;
}

inline __device__ int pos_to_cell_idx(float3 pos) {
    if (pos.x <= par.bounds_min.x || pos.x >= par.bounds_max.x ||
        pos.y <= par.bounds_min.y || pos.y >= par.bounds_max.y ||
        pos.z <= par.bounds_min.z || pos.z >= par.bounds_max.z) {
        return -1;
    }
    return ((int)floorf(pos.z / par.h) * par.gridY  + (int)floorf(pos.y / par.h)) * par.gridX + (int)floorf(pos.x / par.h);
}

__global__ void apply_forces(float3 *velocity, float3 *next_position, float3 *position) {
    __shared__ float3 shared_velocity[APPLY_FORCES_THREADS];
    __shared__ float3 shared_position[APPLY_FORCES_THREADS];
    __shared__ float3 shared_next_position[APPLY_FORCES_THREADS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= par.particleCount) return;

    shared_velocity[threadIdx.x] = velocity[index];
    shared_next_position[threadIdx.x] = next_position[index];
    shared_position[threadIdx.x] = position[index];

    __syncthreads();

    // update velocity and next_postition
    float3 v = par.dt * par.gravity;
    shared_velocity[threadIdx.x] = shared_velocity[threadIdx.x] + v;
    shared_next_position[threadIdx.x] = shared_position[threadIdx.x] + par.dt * shared_velocity[threadIdx.x];

    //check collision
    float3 n = shared_next_position[threadIdx.x];
    if (n.x < par.bounds_min.x) {
        shared_next_position[threadIdx.x].x = par.bounds_min.x + par.dist_from_bound;
        shared_velocity[threadIdx.x].x = 0;
    }
    if (n.x > par.bounds_max.x) {
        shared_next_position[threadIdx.x].x = par.bounds_max.x - par.dist_from_bound;
        shared_velocity[threadIdx.x].x = 0;
    }
    if (n.y < par.bounds_min.y) {
        shared_next_position[threadIdx.x].y = par.bounds_min.y + par.dist_from_bound;
        shared_velocity[threadIdx.x].y = 0;
    }
    if (n.y > par.bounds_max.y) {
        shared_next_position[threadIdx.x].y = par.bounds_max.y - par.dist_from_bound;
        shared_velocity[threadIdx.x].y = 0;
    }
    if (n.z < par.bounds_min.z) {
        shared_next_position[threadIdx.x].z = par.bounds_min.z + par.dist_from_bound;
        shared_velocity[threadIdx.x].z = 0;
    }
    if (n.z > par.bounds_max.z) {
        shared_next_position[threadIdx.x].z = par.bounds_max.z - par.dist_from_bound;
        shared_velocity[threadIdx.x].z = 0;
    }

    // write the the results back to system's arrays
    velocity[index] = shared_velocity[threadIdx.x];
    next_position[index] = shared_next_position[threadIdx.x];
    position[index] = shared_position[threadIdx.x];
}

// find the cell index of each particle, used for finding neighbors
__global__ void cell_map_kernel(int *output, float3 *next_position) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= par.particleCount) return;

    int cell_index = pos_to_cell_idx(next_position[index]);
    output[index] = cell_index;
}

__global__ void get_offset_kernel(int *offsets, int *cell_indices) {
    __shared__ int shared_cell_indices[OFFSET_KERNEL_THREADS + 1];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= par.particleCount) return;

    if (blockIdx.x != 0 && threadIdx.x == 0) {
    shared_cell_indices[threadIdx.x] = cell_indices[index - 1];
    }
    shared_cell_indices[threadIdx.x + 1] = cell_indices[index];

    __syncthreads();

    if (index == 0) {
    offsets[shared_cell_indices[threadIdx.x + 1]] = 0;
    } else if (shared_cell_indices[threadIdx.x] != shared_cell_indices[threadIdx.x + 1]) {
    offsets[shared_cell_indices[threadIdx.x + 1]] = index;
    }
}

void find_neighbors(int gridSize, int particleCount, int *grid_counts, int *grid, int *neighbor_counts, int *neighbors, float3 *next_position, float3 *position, float3 *velocity) {
    int blocks = (particleCount + NUM_THREADS - 1) / NUM_THREADS;

    // Holds cell for given particle index
    cell_map_kernel<<<blocks, NUM_THREADS>>>(neighbor_counts, next_position);
    cudaThreadSynchronize();

    thrust::device_ptr<float3> t_position(position);
    thrust::device_ptr<float3> t_next_position(next_position);
    thrust::device_ptr<float3> t_velocity(velocity);
    thrust::device_ptr<int> keys(neighbor_counts);

    thrust::device_vector<float3> sorted_position(particleCount);
    thrust::device_vector<float3> sorted_next_position(particleCount);
    thrust::device_vector<float3> sorted_velocity(particleCount);

    thrust::counting_iterator<int> iter(0);
    thrust::device_vector<int> indices(particleCount);
    thrust::copy(iter, iter + indices.size(), indices.begin());
    thrust::sort_by_key(keys, keys + particleCount, indices.begin());

    thrust::gather(indices.begin(), indices.end(), t_position, sorted_position.begin());
    thrust::gather(indices.begin(), indices.end(), t_next_position, sorted_next_position.begin());
    thrust::gather(indices.begin(), indices.end(), t_velocity, sorted_velocity.begin());

    thrust::copy(sorted_position.begin(), sorted_position.end(), thrust::raw_pointer_cast(position));
    thrust::copy(sorted_next_position.begin(), sorted_next_position.end(), thrust::raw_pointer_cast(next_position));
    thrust::copy(sorted_velocity.begin(), sorted_velocity.end(), thrust::raw_pointer_cast(velocity));

    cudaMemset(grid_counts, -1, sizeof(int) * gridSize);
    // Grid Counts holds offset into Position for given cell
    int offset_blocks = (particleCount + OFFSET_KERNEL_THREADS - 1) / OFFSET_KERNEL_THREADS;
    get_offset_kernel<<<offset_blocks, OFFSET_KERNEL_THREADS>>>(grid_counts, neighbor_counts);
    cudaThreadSynchronize();

}

__global__ void collision_check(float3 *next_position, float3 *velocity) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= par.particleCount) return;

    float3 n = next_position[index];
    if (n.x < par.bounds_min.x) {
        next_position[index].x = par.bounds_min.x + par.dist_from_bound;
        velocity[index].x = 0;
    }
    if (n.x > par.bounds_max.x) {
        next_position[index].x = par.bounds_max.x - par.dist_from_bound;
        velocity[index].x = 0;
    }
    if (n.y < par.bounds_min.y) {
        next_position[index].y = par.bounds_min.y + par.dist_from_bound;
        velocity[index].y = 0;
    }
    if (n.y > par.bounds_max.y) {
        next_position[index].y = par.bounds_max.y - par.dist_from_bound;
        velocity[index].y = 0;
    }
    if (n.z < par.bounds_min.z) {
        next_position[index].z = par.bounds_min.z + par.dist_from_bound;
        velocity[index].z = 0;
    }
    if (n.z > par.bounds_max.z) {
        next_position[index].z = par.bounds_max.z - par.dist_from_bound;
        velocity[index].z = 0;
    }
}

__device__ float3 get_delta_pos(int *grid_counts, int index, int *neighbor_counts, int *neighbors, float3 *next_position, float *lambda) {
  float w_dq = poly6(par.delta_q * make_vector(1.0f));
  float3 delta_pos = make_vector(0.0f);

  // int neighbor_count = neighbor_counts[index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        float3 p = next_position[index];
        int cell = pos_to_cell_idx(make_float3(p.x + x * par.h, p.y + y * par.h, p.z + z * par.h));
        if (cell < 0) continue;
        int neighbor_index = grid_counts[cell];
        while (true) {
          if (neighbor_index >= par.particleCount) break;
          
          if (neighbor_counts[neighbor_index] != cell) break;

          if (neighbor_index != index) {

            float3 d = next_position[index] - next_position[neighbor_index];

            float kernel_ratio = poly6(d) / w_dq;
            if (w_dq < EPSILON) {
              kernel_ratio = 0.0f;
            }

            float scorr = -par.k * (kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio);
            delta_pos = delta_pos + (lambda[index] + lambda[neighbor_index] + scorr) * spiky_prime(d);

          }

          neighbor_index++;
        }
      }
    }
  }

  return (1.0f / par.rest_density) * delta_pos;
}

__global__ void get_lambda(int *grid_counts, int *neighbor_counts, int *neighbors, float3 *next_position, float *lambda) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= par.particleCount) return;
    float3 p = next_position[index];
    float density_i = 0.0f;
    float ci_gradient = 0.0f;
    float3 accum = make_vector(0.0f);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                int cell = pos_to_cell_idx(make_float3(p.x + x * par.h, p.y + y * par.h, p.z + z * par.h));
                if (cell < 0) continue;
                int neighbor_index = grid_counts[cell];
                // Iterate until out of neighbors
                while (true) {
                    if (neighbor_index >= par.particleCount) break;
                    if (pos_to_cell_idx(next_position[neighbor_index]) != cell) break;
                    // Check we are not at our own particle
                    if (neighbor_index != index) {
                        float3 v = p - next_position[neighbor_index];
                        density_i += poly6(v);
                        float3 sp = spiky_prime(v);
                        ci_gradient += length2(-1.0f / par.rest_density * sp);
                        accum = accum + sp;
                    }
                    neighbor_index++;
                }
            }
        }
    }
    float constraint_i = density_i / par.rest_density - 1.0f;
    ci_gradient += length2((1.0f / par.rest_density) * accum) + par.epsilon;
    lambda[index] = -1.0f * (constraint_i / ci_gradient);
}

__global__ void apply_pressure(int *grid_counts, int *neighbor_counts, int *neighbors, float3 *next_position, float *lambda) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= par.particleCount) return;

  next_position[index] = next_position[index] + get_delta_pos(grid_counts, index, neighbor_counts, neighbors, next_position, lambda);
}

__global__ void apply_viscosity(int *grid_counts, float3 *velocity, float3 *position, float3 *next_position, int *neighbor_counts, int *neighbors) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= par.particleCount) return;

  // Get the viscosity
  float3 viscosity = make_vector(0.0f);

  // int neighbor_count = neighbor_counts[index];
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        float3 p = next_position[index];
        int cell = pos_to_cell_idx(make_float3(p.x + x * par.h, p.y + y * par.h, p.z + z * par.h));
        if (cell < 0) continue;
        int neighbor_index = grid_counts[cell];
        while (true) {
          if (neighbor_index >= par.particleCount) break;
          
          if (neighbor_counts[neighbor_index] != cell) break;

          if (neighbor_index != index) {

            viscosity = viscosity + poly6(position[index] - position[neighbor_index]) * (velocity[index] - velocity[neighbor_index]);
          }

          neighbor_index++;
        }
      }
    }
  }
  velocity[index] = (1.0f / par.dt) * (next_position[index] - position[index]) + par.viscosity_const * viscosity;
  position[index] = next_position[index];
}

void update(int gridSize, int particleCount, int iterations, float3 *velocity, float3 *next_position, float3 *position, int *neighbor_counts, int *neighbors, int *grid_counts, int *grid, float *lambda) {
    int blocks = (particleCount + NUM_THREADS - 1) / NUM_THREADS;
    int force_blocks = (particleCount + APPLY_FORCES_THREADS - 1) / APPLY_FORCES_THREADS;
    apply_forces<<<force_blocks, APPLY_FORCES_THREADS>>>(velocity, next_position, position);
    cudaThreadSynchronize();
    // Clear num_neighbors
    cudaMemset(neighbor_counts, 0, sizeof(int) * particleCount);
    cudaMemset(grid_counts, 0, sizeof(int) * gridSize);
    find_neighbors(gridSize, particleCount, grid_counts, grid, neighbor_counts, neighbors, next_position, position, velocity);
    for (int iter = 0; iter < iterations; iter++) {
        get_lambda<<<blocks, NUM_THREADS>>>(grid_counts, neighbor_counts, neighbors, next_position, lambda);
        cudaThreadSynchronize();
        apply_pressure<<<blocks, NUM_THREADS>>>(grid_counts, neighbor_counts, neighbors, next_position, lambda);
        cudaThreadSynchronize();
        collision_check<<<blocks, NUM_THREADS>>>(next_position, velocity);
        cudaThreadSynchronize();
    }
    apply_viscosity<<<blocks, NUM_THREADS>>>(grid_counts, velocity, position, next_position, neighbor_counts, neighbors);
}

void initialize(struct Parameters *p) {
    cudaCheck(cudaMemcpyToSymbol(par, p, sizeof(struct systempar)));
}
