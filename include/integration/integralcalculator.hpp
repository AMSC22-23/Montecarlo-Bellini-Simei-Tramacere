/**
 * @file integral_calculator.hpp
 * @brief This file contains the declaration of functions related to integral calculation.
 */

#ifndef INTEGRAL_CALCULATOR_HPP
    #define INTEGRAL_CALCULATOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#include "../inputmanager.hpp"  // Assuming this includes necessary headers
#include "geometry/hypersphere.hpp"
#include "geometry/hyperrectangle.hpp"
#include "geometry/hypercube.hpp"
#include "montecarlo.hpp"

/**
 * @brief Factory function to create a geometry object.
 * @details This function creates a geometry object based on the user input.
 * It supports various types of geometries such as hypersphere, hyperrectangle, and hypercube.
 * @param dim The dimensionality of the geometry.
 * @param rad The radius of the geometry (for hypersphere).
 * @param edge The edge length (for hypercube).
 * @param hyper_rectangle_bounds The bounds of the hyperrectangle.
 * @param domain_type The type of the domain.
 * @return A pointer to the created geometry object.
 */
Geometry *geometryFactory(size_t dim, double rad, double edge, std::vector<double> &hyper_rectangle_bounds, std::string domain_type);

/**
 * @brief Core function for integral calculation using the Monte Carlo method.
 * @details This function acts as the core of the vanilla project, embedding multiple methods 
 * for computing the integral using the Monte Carlo method over the selected domain.
 * It reads input from the user to build the integral, then computes it using the Monte Carlo method.
 */
void integralCalculator();

#endif