#include <omp.h>

#include "../include/project/hypersphere.hpp"



double PI = 3.14159265358979323846;

// Constructor
HyperSphere::HyperSphere(const HyperSphere &other) : eng(rd()) {
    dimension = other.dimension;
    radius = other.radius;
    parameter = other.parameter;
    points_inside = other.points_inside;
}

HyperSphere::HyperSphere(int dim, double rad) : eng(rd()) {
    dimension = dim;
    radius = rad;
    parameter = dim / 2.0;
    points_inside = 0;
}


// Generate a random point inside the HyperSphere
void HyperSphere::generate_random_point(std::vector<double> &random_point) {
    std::normal_distribution<double> distribution(-radius, radius);
    random_point.reserve(dimension);

#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
        random_point.push_back(distribution(eng));

    double sum_of_squares = 0.0;
    
#pragma omp parallel for
    for (std::vector<double>::size_type i = 0; i < random_point.size(); ++i)
        sum_of_squares += random_point[i] * random_point[i];

    if (sum_of_squares > radius * radius)
        random_point.clear();
}


// Calculate the volume of the HyperSphere
void HyperSphere::calculate_volume() {
    volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0) * std::pow(radius, dimension);
}


void HyperSphere::add_point_inside() { ++points_inside; }

int HyperSphere::get_points_inside() const { return points_inside; }