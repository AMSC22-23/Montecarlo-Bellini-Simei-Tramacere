#include <omp.h>
#include <iostream>
#include "project/hyperrectangle.hpp"

constexpr double PI = 3.14159265358979323846;

HyperRectangle::HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds) : eng(rd())
{
    dimension = dim;
    parameter = dim / 2.0;
    points_inside = 0;
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

    std::cout << "volume  from class function: " << volume << std::endl;
}

void HyperRectangle::calculate_approximated_volume(int n)
{
    // TODO approximated_volume = (static_cast<double>(points_inside) / n) * hypercube_volume; // Initialize approximated_volume variable
}

void HyperRectangle::add_point_inside() { ++points_inside; }

int HyperRectangle::get_points_inside() const { return points_inside; }

int HyperRectangle::get_dimension() const { return dimension; }

std::pair<double, double> HyperRectangle::Montecarlo_integration(int n, const std::string &function, int dimension)
{
    double total_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(n);

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value)
        for (int i = 0; i < n; ++i)
        {
            this->generate_random_point(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                this->add_point_inside();
            }
        }
    }

    // calculate the integral
    this->calculate_volume();
    double domain = this->get_volume();
    int points_inside = this->get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}
