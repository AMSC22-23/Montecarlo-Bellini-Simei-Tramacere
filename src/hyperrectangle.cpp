#include <omp.h>
#include <iostream>
#include "../include/project/hyperrectangle.hpp"

constexpr double PI = 3.14159265358979323846;

HyperRectangle::HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds) : eng(rd())
{
    dimension = dim;
    this->hyper_rectangle_bounds = hyper_rectangle_bounds;
    volume = 1.0;
}

void HyperRectangle::generate_random_point(std::vector<double> &random_point)
{
    int j = 0;
#pragma omp parallel for
    for (int i = 0; i < dimension * 2 - 1; i += 2)
    {
        std::uniform_real_distribution<double> distribution(hyper_rectangle_bounds[i], hyper_rectangle_bounds[i + 1]);
        random_point[j] = distribution(eng);
        j++;
    }
}

void HyperRectangle::calculate_volume()
{
#pragma omp parallel for
    for (int i = 0; i < 2 * dimension - 1; i += 2)
    {
        volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
    }
}


int HyperRectangle::getdimension() { return dimension; }
