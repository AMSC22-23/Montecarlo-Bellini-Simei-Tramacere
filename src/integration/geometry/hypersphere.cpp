#include "../../../include/integration/geometry/hypersphere.hpp"

  // Constructor
HyperSphere::HyperSphere(size_t dim, double rad)
    :  radius(rad), parameter(dim / 2.0), volume(1.0), dimension(dim), eng(rd()) {}

  // Function to generate a random point in the hypersphere domain
  // for the Monte Carlo method of the original project
void HyperSphere::generateRandomPoint(std::vector<double> &random_point)
{
  bool point_within_sphere = false;
  std::uniform_real_distribution<double> distribution(-radius, radius);

  while (!point_within_sphere)
  {
    double sum_of_squares = 0.0;

    for (size_t i = 0; i < dimension; ++i)
    {
      random_point[i] = distribution(eng);

      sum_of_squares += random_point[i] * random_point[i];
    }

    point_within_sphere = (sum_of_squares <= radius * radius);
  }
}