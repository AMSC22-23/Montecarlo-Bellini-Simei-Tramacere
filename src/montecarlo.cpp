#include <omp.h>
#include <chrono>
#include <utility>

#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"



// Monte Carlo method for the HyperSphere
std::pair<double, double> hs_montecarlo_integration(HyperSphere hs, int n, const std::string &function, int dimension) {
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(dimension);

#pragma omp parallel private(result)
    {
        mu::Parser parser;

#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i) {
            hs.generate_random_point(random_point_vector);
            if (!random_point_vector.empty()) {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
                hs.add_point_inside();
            }
        }
    }

    // calculate the integral
    hs.calculate_volume();
    double domain = hs.get_volume();
    std::cout << std::endl << "Domain: " << domain << std::endl;
    int points_inside = hs.get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}


// Monte Carlo method for the HyperCube
std::pair<double, double> hc_montecarlo_integration(HyperCube hc, int n, const std::string &function, int dimension) {
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(dimension);

#pragma omp parallel private(result)
    {
        mu::Parser parser;

#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i) {
            hc.generate_random_point(random_point_vector);
            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();
            total_value += result;
            total_squared_value += result * result;
        }
    }

    // calculate the integral
    hc.calculate_volume();
    double domain = hc.get_volume();
    std::cout << std::endl << "Domain: " << domain << std::endl;
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;
    
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}


// Monte Carlo method for the HyperRectangle
std::pair<double, double> hr_montecarlo_integration(HyperRectangle hr, int n, const std::string &function, int dimension) {
    double total_value = 0.0;
    double total_squared_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(dimension);

#pragma omp parallel private(result)
    {
        mu::Parser parser;

#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i) {
            hr.generate_random_point(random_point_vector);
            if (!random_point_vector.empty()) {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                total_squared_value += result * result;
            }
        }
    }

    // calculate the integral
    hr.calculate_volume();
    double domain = hr.get_volume();
    std::cout << std::endl << "Domain: " << domain << std::endl;
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;
    
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}
