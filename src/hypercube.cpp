#include <omp.h>
#include <iostream>
#include "project/hypercube.hpp"

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

std::pair<double, double> HyperCube::Montecarlo_integration(int n, const std::string &function)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(this->dimension);

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            this->generate_random_point(random_point_vector);
            
            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();
            total_value += result;
            total_squared_value += result * result;
        }
    }

    // calculate the integral
    this->calculate_volume();
    double domain = this->get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;
    
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}
