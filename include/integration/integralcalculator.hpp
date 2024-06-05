#ifndef INTEGRAL_CALCULATOR_HPP
    #define INTEGRAL_CALCULATOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#include "../inputmanager.hpp"
#include "geometry/hypersphere.hpp"
#include "geometry/hyperrectangle.hpp"
#include "geometry/hypercube.hpp"
#include "montecarlo.hpp"

  /**
 * @brief This function is used to create a geometry object
 * @details The function creates a geometry object based on the user input
 * @param dim An integer representing the dimension of the geometry
 * @param rad A double representing the radius of the geometry
 * @param edge A double representing the edge of the geometry
 * @param hyper_rectangle_bounds A vector of doubles representing the bounds of the hyperrectangle
 * @param domain_type A string representing the type of the domain
 * @return A pointer to the geometry object
 */
Geometry *geometryFactory(size_t dim, double rad, double edge, std::vector<double> &hyper_rectangle_bounds, std::string domain_type);

  /**
 * @brief This function if the core of the vanilla project: 
 * it embeds multiple methods that are used to compute
 * the integral using the Monte Carlo method over the selected domain
 * @details The function reads the input from the user and builds the integral,
 * then it computes the integral using the Monte Carlo method.
 */
void integralCalculator();

#endif