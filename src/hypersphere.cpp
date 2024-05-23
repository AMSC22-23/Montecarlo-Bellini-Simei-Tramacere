#include <omp.h>
#include "../include/project/hypersphere.hpp"

constexpr double PI = 3.14159265358979323846;

HyperSphere::HyperSphere(int dim, double rad) : radius(rad), parameter(dim / 2.0), volume(0.0), dimension(dim) {}


void HyperSphere::generateRandomPoint(std::vector<double> &random_point)
{
    std::vector<double> local_random_point(dimension);
    double local_sum_of_squares = 0.0;
    omp_set_num_threads(random_point.size());

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_local std::default_random_engine eng(rd() + thread_id);
        std::uniform_real_distribution<double> distribution(-radius, radius);

#pragma omp for reduction(+ : local_sum_of_squares)
        for (size_t i = 0; i < dimension; ++i)
        {
            local_random_point[i] = distribution(eng);
            local_sum_of_squares += local_random_point[i] * local_random_point[i];
        }
    }

    if (local_sum_of_squares <= radius * radius)
    {
        std::copy(local_random_point.begin(), local_random_point.end(), random_point.begin());
    }
    else
    {
        random_point[0] = 0.0;
    }
}

void HyperSphere::calculateVolume()
{
    volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0) * std::pow(radius, dimension);
}