#include "Serial_System.h"
#include <random>

//Initializes the system and generates initial particles according to the input(nums, bound, configuration)
SerialSystem::SerialSystem(unsigned numParticles, glm::vec3 bounds_max, std::string config): System(numParticles, bounds_max), neighbors_map(5){
    //Set the grid
    max_x = size_t(ceil((bounds_max.x - bounds_min.x) / h));
    max_y = size_t(ceil((bounds_max.y - bounds_min.y) / h));
    max_z = size_t(ceil((bounds_max.z - bounds_min.z) / h));
    particlePos.resize(numParticles);
    particles.resize(numParticles);

    // Generate particles based on the specified configuration
    std::default_random_engine generator;
    if (config == "dam") {
        std::uniform_real_distribution<float> distributionX(bounds_min.x + 0.1f, bounds_min.x + bounds_max.x * 0.9f);
        std::uniform_real_distribution<float> distributionY(bounds_min.y + 0.1f, bounds_max.y - 0.1f);
        std::uniform_real_distribution<float> distributionZ(bounds_min.z + bounds_max.z * 0.9f, bounds_max.z - 0.1f);
        for (int i = 0; i < numParticles; i++) {
            particles[i] = (i % 2 == 0)? new Particle(glm::vec3(distributionX(generator), distributionY(generator), distributionZ(generator)), maxNeighbors):
            particles[i] = new Particle(glm::vec3(distributionZ(generator), distributionY(generator), distributionX(generator)), maxNeighbors);
            particlePos[i] = particles[i]->position;
        }
    }
    else if (config == "sphere") {
        float r = std::min(std::min(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y), bounds_max.z - bounds_min.z) / 2.0;
        glm::vec3 offset = glm::vec3((bounds_max.x - bounds_min.x) / 2.0, (bounds_max.y - bounds_min.y) / 2.0, (bounds_max.z - bounds_min.z) / 2.0);
        std::uniform_real_distribution<float> distributionR(-r, r);
        float x, y, z;
        for (int i = 0; i < numParticles; i++) {
            do {
                x = distributionR(generator);
                y = distributionR(generator);
                z = distributionR(generator);
            } while (x * x + y * y + z * z >= r * r);
        particles[i] = new Particle(glm::vec3(x, y, z) + offset, maxNeighbors);
        particlePos[i] = particles[i]->position;
        }
    } 
    else {
        std::uniform_real_distribution<float> distribution(bounds_min.x + 5, bounds_max.x - 5);
        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle( glm::vec3(distribution(generator), distribution(generator), distribution(generator)), maxNeighbors);
            particlePos[i] = particles[i]->position;
        }
    }
    std::cout << particles.size() << " particles generated!" << std::endl;
}

SerialSystem::~SerialSystem() {for (auto i : particles) delete (i);}

// the Poly6 kernel, used for smoothing and density calculation
double SerialSystem::poly6(glm::vec3 r) {
    double norm_coeff = (h * h - glm::dot(r, r));
    if (norm_coeff <= 0) {
        return 0.0;
    }
    if (r.x == 0.0f && r.y == 0.0f && r.z == 0.0f) {
        return 0.0;
    }
    return poly6_const * norm_coeff * norm_coeff * norm_coeff;
}

// gradient of the Spiky kernel, used for calculating forces
glm::vec3 SerialSystem::spiky_prime(glm::vec3 r) {
    glm::vec3 r_norm = glm::normalize(r);
    double norm_coeff = (h - glm::l2Norm(r));
    if (norm_coeff <= 0) {
        return glm::vec3(0.0f);
    }
    if (r.x == 0.0f && r.y == 0.0f && r.z == 0.0f) {
        return glm::vec3(0.0f);
    }
    return spiky_const * norm_coeff * norm_coeff * r_norm;
}

// Calculate next position under the force of gravity. Need to be optimized later.
void SerialSystem::apply_forces() {
  for (int i = 0; i < numParticles; i++) {
    auto &p = particles[i];
    p->velocity += dt * gravity;
    p->next_position = p->position + dt * p->velocity;
    p->boundary = false;
  }
}

// Find neighbors for each particle, add the neighbor to particle->neighbors.
void SerialSystem::find_neighbors() {
  neighbors_map.clear();
  //neighbors_map: [pos/h: particle] contains particles in each cell
  for (auto &p : particles) neighbors_map.emplace(std::make_tuple(floor(p->next_position.x / h), floor(p->next_position.y / h), floor(p->next_position.z / h)), p);
  for (int i = 0; i < numParticles; i++) {
    auto &p = particles[i];
    p->neighbors.clear();
    //The bounding box contains all possible grid cells that is adjacent to p
    glm::vec3 BB_min = p->next_position - glm::vec3(h, h, h);
    glm::vec3 BB_max = p->next_position + glm::vec3(h, h, h);

    //For each possible adjacent cell, iterate over all particles to see if they are in that cell 
    for (double x = BB_min.x; x <= BB_max.x; x += h) {
      for (double y = BB_min.y; y <= BB_max.y; y += h) {
        for (double z = BB_min.z; z <= BB_max.z; z += h) {
          auto range = neighbors_map.equal_range(std::make_tuple(floor(x / h), floor(y / h), floor(z / h))); //return an iterator that covers the elements has the same key
          if (range.first == range.second) continue;
          for (auto it = range.first; it != range.second; ++it) {
            Particle *j = it->second;
            if (j != p) {
              double length = glm::l2Norm(p->next_position, j->next_position);
              //if the Euclidean distance is within h, add to p's neighbors
              if (length < h) p->neighbors.push_back(j);
            }
          }
        }
      }
    }
  }
}

// Calculate the density at a grid cell by apply poly6() to the particles in that cell
double SerialSystem::calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex) {
    double scalar = 0.0f;
    auto range = neighbors_map.equal_range(std::make_tuple(i, j, k));
    if (range.first == range.second) return 0.0f;
    for (auto it = range.first; it != range.second; ++it) {
        Particle *p = it->second;
        double length = glm::l2Norm(grid_vertex, p->next_position);
        if (length < h) scalar += poly6(grid_vertex - p->next_position);
    }
    return scalar;
}

// Calculate the scalar field value by adding the cell and its the neighbors contribution
double SerialSystem::calc_scalar(size_t i, size_t j, size_t k) {
    double scalar = 0.0f;
    glm::vec3 grid_vertex(i * h, j * h, k * h);
    scalar += calc_cell_density(i, j, k, grid_vertex);
    scalar += calc_cell_density(i - 1, j, k, grid_vertex);
    scalar += calc_cell_density(i, j - 1, k, grid_vertex);
    scalar += calc_cell_density(i - 1, j - 1, k, grid_vertex);
    scalar += calc_cell_density(i, j, k - 1, grid_vertex);
    scalar += calc_cell_density(i - 1, j, k - 1, grid_vertex);
    scalar += calc_cell_density(i, j - 1, k - 1, grid_vertex);
    scalar += calc_cell_density(i - 1, j - 1, k - 1, grid_vertex);
    return scalar;
}

// Compute the lambda value for correcting the paricles' position
void SerialSystem::get_lambda() {
    for (auto i : particles) {
        //1.Calulate the each particle's next position's density
        double density_i = 0.0f;
        for (auto j : i->neighbors) {
            density_i += poly6(i->next_position - j->next_position);
        }
        i->density = density_i;
        //2. Calculate each particle's density constraint and spiky
        double constraint_i = density_i / rest_density - 1.0f;
        double ci_gradient = 0.0f;
        for (auto j : i->neighbors) ci_gradient += glm::length2(-1.0f / rest_density * spiky_prime(i->next_position - j->next_position));
        glm::vec3 accum = glm::vec3(0.0f);
        for (auto j : i->neighbors) accum += spiky_prime(i->next_position - j->next_position);
        ci_gradient += glm::length2((1.0f / rest_density) * accum);
        ci_gradient += epsilon;
        i->lambda = -1.0f * (constraint_i / ci_gradient);
    }
}

// Compute the position correction for a particle based on its neighbors and the pressure forces.
glm::vec3 SerialSystem::get_delta_pos(Particle *i) {
  double w_dq = poly6(delta_q * glm::vec3(1.0f)); //reference value at distance = delta_q
  glm::vec3 delta_pos(0.0f);
  for (auto j : i->neighbors) {
    double kernel_ratio = poly6(i->next_position - j->next_position) / w_dq;
    if (w_dq < glm::epsilon<double>()) {
      kernel_ratio = 0.0f;
    }
    double scorr = -k * (kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio * kernel_ratio);
    delta_pos +=
        (i->lambda + j->lambda + scorr) * spiky_prime(i->next_position - j->next_position);
  }

  return (1.0f / rest_density) * delta_pos;
}

void SerialSystem::collision_check(Particle *i) {
  if (i->next_position.x < bounds_min.x) {
      i->next_position.x = bounds_min.x + dist_from_bound;
      i->boundary = true;
      i->velocity.x = 0;
  }
  if (i->next_position.x > bounds_max.x) {
      i->next_position.x = bounds_max.x - dist_from_bound;
      i->boundary = true;
      i->velocity.x = 0;
  }
  if (i->next_position.y < bounds_min.y) {
      i->next_position.y = bounds_min.y + dist_from_bound;
      i->boundary = true;
      i->velocity.y = 0;
  }
  if (i->next_position.y > bounds_max.y) {
      i->next_position.y = bounds_max.y - dist_from_bound;
      i->boundary = true;
      i->velocity.y = 0;
  }
  if (i->next_position.z < bounds_min.z) {
      i->next_position.z = bounds_min.z + dist_from_bound;
      i->boundary = true;
      i->velocity.z = 0;
  }
  if (i->next_position.z > bounds_max.z) {
      i->next_position.z = bounds_max.z - dist_from_bound;
      i->boundary = true;
      i->velocity.z = 0;
  }
}

// Apply pressure forces to correct particle positions after the lambda values are computed.
void SerialSystem::apply_pressure() {
  for (auto i : particles) {
      glm::vec3 dp = get_delta_pos(i);
      i->next_position += dp;
      collision_check(i);
  }
}

// Compute the viscosity effect on a particle's velocity based on its neighbors' velocities
glm::vec3 SerialSystem::get_viscosity(Particle *i) {
  glm::vec3 visc = glm::vec3(0.0f);

  for (auto j : i->neighbors) {
    visc += (i->velocity - j->velocity) * poly6(i->position - j->position);
  }
  return viscosity_const * visc;
}

// Step the simulation
void SerialSystem::step() {
  apply_forces();

  find_neighbors();

  for (int iter = 0; iter < iterations; iter++) {
    get_lambda();
    apply_pressure();
  }

  for (auto &i : particles) {
    i->velocity = (1.0f / dt) * (i->next_position - i->position);
    i->velocity += get_viscosity(i);
    i->position = i->next_position;
  }

  for (int i = 0; i < particles.size(); i++) {
    particlePos[i] = particles[i]->position;
  }
}
