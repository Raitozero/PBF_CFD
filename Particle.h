#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>
#include <vector>
using namespace std;

class Particle {
 public:
  glm::vec3 position, next_position;
  glm::vec3 velocity;
  vector<Particle*> neighbors;
  double lambda, density; // Lambda -> fluid constraints, density -> particle's density
  bool boundary, surface;

  Particle(glm::vec3 pos, size_t maxNeighbors): 
    position(pos), velocity(0, 0, 0),
    next_position(0, 0, 0), lambda(0.0),  density(0.0),
    surface(false), boundary(false) 
    {neighbors.reserve(maxNeighbors);}
};

#endif