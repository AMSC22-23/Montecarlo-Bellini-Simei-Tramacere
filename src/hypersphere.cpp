#include <omp.h>
#include "../include/project/hypersphere.hpp"

constexpr double PI = 3.14159265358979323846;

HyperSphere::HyperSphere(int dim, double rad) : eng(rd())
{
    dimension = dim;
    radius = rad;
    parameter = dim / 2.0;
    points_inside = 0;
}

void HyperSphere::generate_random_point(std::vector<double> &random_point)
{
    std::uniform_real_distribution<double> distribution(-radius, radius);
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

void HyperSphere::calculate_volume()
{
    volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0) * std::pow(radius, dimension);
}

std::pair<double, double> HyperSphere::Montecarlo_integration(int n, const std::string &function, int dimension)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(dimension);
#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            this->generate_random_point(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
                this->add_point_inside();
            }
        }
    }

    // calculate the integral
    this->calculate_volume();
    double domain = this->get_volume();
    std::cout << "domain: " << domain << std::endl;
    int points_inside = this->get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

void HyperSphere::add_point_inside() { ++points_inside; }

int HyperSphere::get_points_inside() const { return points_inside; }
