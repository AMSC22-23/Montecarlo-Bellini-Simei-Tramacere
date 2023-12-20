#include <random>
#include <vector>
#include <cmath>


#include "/usr/local/opt/libomp/include/omp.h"
#include "project/hypersphere.hpp"


constexpr double PI = 3.14159265358979323846;


HyperSphere::HyperSphere(int dim, double rad) : eng(rd()) {
    dimension = dim;
    radius = rad;
    parameter = dim / 2.0;
    hypercube_volume = pow(2 * radius, dimension);
    points_inside = 0;
}


void HyperSphere::generate_random_point(std::vector<double> &random_point) {
    random_point.resize(dimension);

    #pragma omp parallel
    std::default_random_engine local_eng(rd());
    std::uniform_real_distribution<double> distribution(-radius, radius);

    #pragma omp for
    for (int i = 0; i < dimension; ++i) {
        random_point[i] = distribution(local_eng);
    }

    // Check if the point is inside the hypersphere
    double sum_of_squares = 0.0;
    double x = 0.0;
    int size = random_point.size();

    #pragma omp parallel for reduction(+ : sum_of_squares)
    for (int i = 0; i < size; ++i) {
        x = random_point[i];
        sum_of_squares += x * x;
    }

    if (sum_of_squares > radius * radius) random_point.clear();
}


void HyperSphere::calculate_volume() {
    volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0) * std::pow(radius, dimension);
}

void HyperSphere::calculate_approximated_volume(int n) {
    approximated_volume = (static_cast<double>(points_inside) / n) * hypercube_volume;
}


void HyperSphere::add_point_inside() { ++points_inside; }


int HyperSphere::get_points_inside() const { return points_inside; }


int HyperSphere::get_dimension() const { return dimension; }