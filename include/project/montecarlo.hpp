#ifndef PROJECT_MONTECARLOINTEGRATOR_
    #define PROJECT_MONTECARLOINTEGRATOR_

#include <omp.h>
#include <iostream>

#include "hypercube.hpp"
#include "hyperrectangle.hpp"
#include "hypersphere.hpp"
#include "asset.hpp"


/**
 * @brief Compute the integral using the Monte Carlo method for a hypersphere
 * @details This function computes the integral using the Monte Carlo method.
 * @param n The number of points
 * @param function The function to integrate
 * @param hypercube The hypercube
 * @param variance The variance
 * @return A pair containing the integral and the standard error
 */
std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperCube &hypercube,
                                                double &variance);

/**
 * @brief Compute the integral using the Monte Carlo method for a hyperrectangle
 * @details This function computes the integral using the Monte Carlo method.
 * @param n The number of points
 * @param function The function to integrate
 * @param hyperrectangle The hyperrectangle
 * @param variance The variance
 * @return A pair containing the integral and the standard error
 */
std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperRectangle &hyperrectangle,
                                                double &variance);

/**
 * @brief Compute the integral using the Monte Carlo method for a hypersphere
 * @details This function computes the integral using the Monte Carlo method.
 * @param n The number of points
 * @param function The function to integrate
 * @param hypersphere The hypersphere
 * @param variance The variance
 * @return A pair containing the integral and the standard error
 */
std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperSphere &hypersphere,
                                                double &variance);


#endif