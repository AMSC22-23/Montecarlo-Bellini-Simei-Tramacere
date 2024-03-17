#include <omp.h>
#include <iostream>
#include "../include/project/hyperrectangle.hpp"



// Constructor
HyperRectangle::HyperRectangle(const HyperRectangle &other) : eng(rd()) {
    dimension = other.dimension;
    hyper_rectangle_bounds = other.hyper_rectangle_bounds;
    volume = other.volume;
}

HyperRectangle::HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds) : eng(rd()) {
    dimension = dim;
    this->hyper_rectangle_bounds = hyper_rectangle_bounds;
    volume = 1.0;
}


// Generate a random point inside the HyperRectangle
void HyperRectangle::generate_random_point(std::vector<double> &random_point) {
    int j = 0;

#pragma omp parallel for
    for (int i = 0; i < dimension * 2 - 1; i += 2) {
        std::normal_distribution<double> distribution(hyper_rectangle_bounds[i], hyper_rectangle_bounds[i + 1]);
        random_point[j] = distribution(eng);
        j++;
    }
}


// Calculate the volume of the HyperRectangle
void HyperRectangle::calculate_volume() {
    
#pragma omp parallel for
    for (int i = 0; i < 2 * dimension - 1; i += 2) {
        volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
    }
}
