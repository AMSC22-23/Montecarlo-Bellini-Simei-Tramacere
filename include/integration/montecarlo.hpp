#ifndef PROJECT_MONTECARLOINTEGRATOR_
    #define PROJECT_MONTECARLOINTEGRATOR_

#include <omp.h>
#include <iostream>

#include "geometry/hypercube.hpp"
#include "geometry/hyperrectangle.hpp"
#include "geometry/hypersphere.hpp"
#include "../optionpricing/asset.hpp"

    /**
 * @brief Compute the integral using the Monte Carlo method for a generic domain.
 * @details This function computes the integral using the Monte Carlo method for a generic domain.
 * @tparam DomainType The type of domain object (e.g., HyperCube, HyperRectangle, HyperSphere)
 * @param n The number of points
 * @param function The function to integrate
 * @param domain The domain object
 * @param variance The variance
 * @return A pair containing the integral and the standard error
 */
template <typename DomainType>
std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                DomainType &domain,
                                                double &variance)
{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    mu::Parser parser;

    std::vector<double> random_point_vector;
    random_point_vector.resize(domain.getDimension());

    std::cout << "Computing integral..." << std::endl;

        // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

        // The Monte Carlo method is parallelized using OpenMP to speed up the computation
#pragma omp parallel private(parser, result)
    {
            // Thread-local accumulation variables
        double local_total_value         = 0.0;
        double local_total_squared_value = 0.0;
        std::vector<double> local_random_point_vector(domain.getDimension());

#pragma omp for reduction(+ : total_value, total_squared_value) schedule(dynamic)
        for (size_t i = 0; i < n; ++i)
        {
            domain.generateRandomPoint(local_random_point_vector);

            result = evaluateFunction(function, local_random_point_vector, parser);
            parser.ClearVar();

            local_total_value         += result;
            local_total_squared_value += result * result;
        }

#pragma omp critical
        {
            total_value         += local_total_value;
            total_squared_value += local_total_squared_value;
        }
    }

        // Calculate the integral
    domain.calculateVolume();
    double volume   = domain.getVolume();
    double integral = total_value / static_cast<double>(n) * volume;

        // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

        // Calculate the variance
    variance = (total_squared_value / static_cast<double>(n)) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));

        // Compute time taken
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, static_cast<double>(duration.count()));
}

#endif
