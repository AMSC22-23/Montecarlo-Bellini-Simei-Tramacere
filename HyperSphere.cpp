#include <random>
#include <vector>
#include <cmath>
#include <omp.h>
#include "HyperSphere.hpp"


    HyperSphere::HyperSphere(int dim, double rad) : eng(rd())
    {
        dimension = dim;
        radius = rad;
        parameter = dim / 2.0;
        hypercube_volume = pow(2 * radius, dimension);
        points_inside = 0;
    }

    void HyperSphere::generate_random_point(std::vector<double> &random_point)
    {
        std::uniform_real_distribution<double> distribution(-radius, radius);
        random_point.reserve(dimension);

#pragma omp parallel for
        for (int i = 0; i < dimension; ++i)
        {
            random_point.push_back(distribution(eng));
        }
        
        // check if the point is inside the hypersphere
            //@note: here a for loop would have been more effective
            //@note: power with integer exponent should not use `std::pow`
        double sum_of_squares = std::accumulate(random_point.begin(), random_point.end(),
                                                0.0, [](double sum, double x)
                                                { return sum + std::pow(x, 2); });
        if (sum_of_squares > std::pow(radius, 2))
            random_point.clear();
    }

    void HyperSphere::calculate_volume()
    {
            //@note: should use std:: for math functions
            //@note: the number header for \pi instead of M_PI
        volume = pow(M_PI, parameter) / tgamma(parameter + 1.0) * pow(radius, dimension);
    }

    void HyperSphere::calculate_approximated_volume(int n)
    {
        approximated_volume = (static_cast<double>(points_inside) / n) * hypercube_volume;
    }

    void HyperSphere::add_point_inside() { ++points_inside; }

    int HyperSphere::get_points_inside() const { return points_inside; }

    int HyperSphere::get_dimension() const { return dimension; }

