#define GLM_ENABLE_EXPERIMENTAL

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "Particle.h"
#include "System.h"

// Hash grid coordinates for neighbor search.
static size_t MyHash(const std::tuple<size_t, size_t, size_t> &k) {
  const size_t prime1 = 2654435761;
  const size_t prime2 = 805306457;
  const size_t prime3 = 4294967311;
  return ((std::get<0>(k) * prime1) + (std::get<1>(k) * prime2) + (std::get<2>(k) * prime3)) % 100003;
}

static bool HashEqual(const std::tuple<size_t, size_t, size_t> &t1, const std::tuple<size_t, size_t, size_t> &t2) {
    return (std::get<0>(t1) == std::get<0>(t2) && std::get<1>(t1) == std::get<1>(t2) && std::get<2>(t1) == std::get<2>(t2));
    }

//hashmap [coordinates:particle]
typedef std::unordered_multimap<std::tuple<size_t, size_t, size_t>, Particle *,decltype(MyHash), decltype(HashEqual)>hashMap;

class SerialSystem : public System {
 public:
  SerialSystem(unsigned numParticles, glm::vec3 bounds_max, std::string config);
  float *getParticlePos(); //return the pointer to particlePos[0].x
  unsigned getParticleNum() { return numParticles; };
  void step();
  virtual ~SerialSystem();

 private:
  std::vector<glm::vec3> particlePos;
  size_t max_x, max_y, max_z;  // Grid dimensions
  std::vector<Particle*> particles; 
  std::vector<double> scalar_field; // Scalar field for fluid calculations
  hashMap neighbors_map;  // Hash map for storing particle neighbors
  // Kernel function for calculating density contribution from a distance vector r
  double poly6(glm::vec3 r);
  // Kernel function for calculating gradient of the Spiky kernel
  glm::vec3 spiky_prime(glm::vec3 r);
  void apply_forces();
  void find_neighbors();
  double calc_cell_density(size_t i, size_t j, size_t k, glm::vec3 grid_vertex);
  double calc_scalar(size_t i, size_t j, size_t k);
  void get_lambda();
  glm::vec3 get_delta_pos(Particle *i);
  void collision_check(Particle *i);
  void apply_pressure();
  glm::vec3 get_viscosity(Particle *i);
};
