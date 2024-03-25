#include "../include/project/montecarlointegrator.hpp"

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperCube &hypercube)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(hypercube.getdimension());

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            hypercube.generate_random_point(random_point_vector);
            
            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();
            total_value += result;
            total_squared_value += result * result;
        }
    }

    // calculate the integral
    hypercube.calculate_volume();
    double domain = hypercube.get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;
    
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperRectangle &hyperrectangle)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(hyperrectangle.getdimension());

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            hyperrectangle.generate_random_point(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
            }
        }
    }

    // calculate the integral
    hyperrectangle.calculate_volume();
    double domain = hyperrectangle.get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;
    
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperSphere &hypersphere)

{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(hypersphere.getdimension());
// #pragma omp parallel private(result)
    {
        mu::Parser parser;
// #pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            hypersphere.generate_random_point(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
                hypersphere.add_point_inside();
            }
        }
    }

    // calculate the integral
    hypersphere.calculate_volume();
    double domain = hypersphere.get_volume();
    std::cout << "domain: " << domain << std::endl;
    int points_inside = hypersphere.get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}