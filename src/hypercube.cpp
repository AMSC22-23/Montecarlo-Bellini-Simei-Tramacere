#include <omp.h>
#include <iostream>
#include "../include/project/hypercube.hpp"

HyperCube::HyperCube(int dim, double edge)
    : edge(edge), dimension(dim), volume(1.0), eng(rd()) {}

void HyperCube::generateRandomPoint(std::vector<double>& random_point)
{
    std::uniform_real_distribution<double> distribution(-edge / 2, edge / 2);

#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
        random_point[i] = distribution(eng);
}

void HyperCube::calculateVolume()
{
    for (int i = 0; i < dimension; ++i)
    {
        volume *= edge;
    }
}