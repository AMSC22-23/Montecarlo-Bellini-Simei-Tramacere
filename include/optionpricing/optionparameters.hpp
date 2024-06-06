/**
 * @file option_parameters.hpp
 * @brief This file contains declarations related to option parameters.
 */

#ifndef OPTION_PARAMETERS_
    #define OPTION_PARAMETERS_

#include <vector>
#include <string>
#include <cmath>

#include "asset.hpp"

  /**
 * @brief This function creates the payoff function for the option
 * @details The function computes the payoff function for the option
 * by taking the sum of the last real value of each asset and multiplying it by a vector of unknowns
 * @param k The strike price
 * @param assets The assets
 * @return A pair containing a string representing the payoff function and a vector of coefficients
 * which are the last real values of the assets
 */
std::pair<std::string, std::vector<double>> createPayoffFunction(double k, const std::vector<Asset> &assets);

  /**
 * @brief This function calculates the strike price for the option
 * @details The function calculates the strike price of the option
 * by taking the sum of the last real value of each asset
 * @param assets The assets
 * @return A double representing the strike price
 */
double calculateStrikePrice(const std::vector<Asset> &assets);

#endif