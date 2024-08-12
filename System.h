#ifndef SYSTEM_H
#define SYSTEM_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include "Particle.h"

class System {
public:
    virtual unsigned getParticleNum() = 0;
    virtual float* getParticlePos() = 0;
    virtual void step() = 0;

protected:
    System(unsigned numParticles, glm::vec3 bounds_max): numParticles(numParticles), bounds_max(bounds_max) {}
    const unsigned numParticles;
    const size_t maxNeighbors = 50;
    const glm::vec3 gravity = glm::vec3(0.0, -9.8, 0.0);
    glm::vec3 bounds_min = glm::vec3(0.0, 0.0, 0.0);
    glm::vec3 bounds_max;
    const int iterations = 10;
    const double dt = 0.02; //unit time
    const double h = 1.5; //smoothing length
    const double rest_density = 2000;
    const double epsilon = 0.01; // Small value to avoid division by zero
    const double k = 0.01; // Constant for delta_pos correction
    const double delta_q = 0.2 * h; // Distance for calculating pressure correction
    const double dist_from_bound = 0.0001; // Distance from the boundary to check for collisions
    const double viscosity_const = 0.1;
    const double poly6_const = 315.f / (64.f * glm::pi<double>() * h * h * h * h * h * h * h * h * h);
    const double spiky_const = 45.f / (glm::pi<double>() * h * h * h * h * h * h);
};

#endif