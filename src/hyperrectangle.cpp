#include "../include/project/hyperrectangle.hpp"


// Constructor
HyperRectangle::HyperRectangle(size_t dim, std::vector<double> &hyper_rectangle_bounds)
    : hyper_rectangle_bounds(hyper_rectangle_bounds), volume(1.0), dimension(dim), eng(rd()) {}

// Function to generate a random point in the hyperrectangle domain
// for the Monte Carlo method of the original project
void HyperRectangle::generateRandomPoint(std::vector<double> &random_point)
{
    int j = 0;
    for (size_t i = 0; i < dimension * 2 - 1; i += 2)
    {
        // Generate random points by following the uniform distribution
        std::uniform_real_distribution<double> distribution(hyper_rectangle_bounds[i], hyper_rectangle_bounds[i + 1]);
        random_point[j] = distribution(eng);
        j++;
    }
}
