#include <omp.h>
#include <iostream>
#include "../include/project/hypercube.hpp"



// Constructor
HyperCube::HyperCube(const HyperCube &other) : eng(rd()) {
    dimension = other.dimension;
    edge = other.edge;
    volume = other.volume;
}

HyperCube::HyperCube(int dim, double &edge) : eng(rd()) {
    dimension = dim;
    this->edge = edge;
    volume = 1.0;
}


// Generate a random point inside the HyperCube
void HyperCube::generate_random_point(std::vector<double> &random_point) {
    std::normal_distribution<double> distribution(-edge / 2, edge / 2);

#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
        random_point[i] = (distribution(eng));
}


// Calculate the volume of the HyperCube
void HyperCube::calculate_volume() {
    this->volume = 1.0;
    for (int i = 0; i < dimension; ++i) {
        this->volume *= this->edge;
    }
    std::cout << "Volume of the hypercube: " << this->volume << std::endl;
}