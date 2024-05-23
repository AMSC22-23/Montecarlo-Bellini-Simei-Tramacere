#include "../include/project/montecarlo.hpp"
#include "../include/project/functionevaluator.hpp"

std::pair<double, double> montecarloIntegration(int n, const std::string &function, HyperCube &hypercube, double &variance)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    std::vector<double> random_point_vector(hypercube.getDimension());

    std::cout << "Computing integral..." << std::endl;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < n; ++i)
        {
            hypercube.generateRandomPoint(random_point_vector);

            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();

            total_value += result;
            total_squared_value += result * result;
        }
    }

    // calculate the integral
    hypercube.calculateVolume();
    double domain = hypercube.getVolume();
    double integral = total_value / static_cast<double>(n) * domain;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // calculate the variance
    variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));

    // compute time taken
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

std::pair<double, double> montecarloIntegration(int n, const std::string &function, HyperRectangle &hyperrectangle, double &variance)
{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    std::vector<double> random_point_vector(hyperrectangle.getDimension());

    std::cout << "Computing integral..." << std::endl;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel private(result)
    {
        mu::Parser parser;

#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < n; ++i)
        {
            hyperrectangle.generateRandomPoint(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
            }
            else
            {
                std::cout << "Error generating random point" << std::endl;
                i--;
            }
        }
    }

    // calculate the integral
    hyperrectangle.calculateVolume();
    double domain = hyperrectangle.getVolume();
    double integral = total_value / static_cast<double>(n) * domain;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // calculate the variance
    variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));

    // compute time taken
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, static_cast<double>(duration.count()));
}

std::pair<double, double> montecarloIntegration(int n, const std::string &function, HyperSphere &hypersphere, double &variance)

{
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    std::vector<double> random_point_vector(hypersphere.getDimension());

    std::cout << "Computing integral..." << std::endl;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    
#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < n; ++i)
        {
            hypersphere.generateRandomPoint(random_point_vector);

            if (random_point_vector[0] != 0.0)
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
            }
            else
            {
                i--;
            }
        }
    }

    // calculate the integral
    hypersphere.calculateVolume();
    double domain = hypersphere.getVolume();
    double integral = total_value / static_cast<double>(n) * domain;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // calculate the variance
    variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));

    // compute time taken
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}
