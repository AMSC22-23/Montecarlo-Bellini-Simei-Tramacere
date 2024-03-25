#include <omp.h>
#include <iostream>
#include "../include/project/hypercube.hpp"

HyperCube::HyperCube(int dim, double &edge) : eng(rd())
{
    dimension = dim;
    this->edge = edge;
    volume = 1.0;
}

void HyperCube::generate_random_point(std::vector<double> &random_point)
{
    std::uniform_real_distribution<double> distribution(-edge / 2, edge / 2);

#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
        random_point[i] = (distribution(eng));
}

void HyperCube::calculate_volume()
{
    this->volume = 1.0;
    for (int i = 0; i < dimension; ++i)
    {
        this->volume *= this->edge;
    }
    std::cout << "Volume of the hypercube: " << this->volume << std::endl;
}

    int HyperCube::getdimension() { return dimension; }