#include "../../../include/integration/geometry/hypercube.hpp"

  // Constructor
HyperCube::HyperCube(size_t dim, double edge)
    :  edge(edge), dimension(dim), volume(1.0), eng(rd()) {}

  // Functions that generate a random point in the hypercube
  // by parallelizing the generation for each dimension
  // This function is used in the Monte Carlo method of the original project
void HyperCube::generateRandomPoint(std::vector<double> &random_point)
{
    std::uniform_real_distribution<double> distribution(-edge / 2, edge / 2);

#pragma omp parallel for
    for (size_t i = 0; i < dimension; ++i)
        random_point[i] = distribution(eng);
}