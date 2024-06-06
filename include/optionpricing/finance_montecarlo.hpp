/**
 * @file finance_montecarlo.hpp
 * @brief This file contains declarations related to the core functions of the pricing.
 */

#ifndef PROJECT_FINANCEMONTECARLO_HPP
    #define PROJECT_FINANCEMONTECARLO_HPP

#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <random>

#include "finance_inputmanager.hpp"
#include "../integration/geometry/hyperrectangle.hpp"
#include "asset.hpp"
#include "finance_enums.hpp"
#include "../../include/optionpricing/finance_montecarloutils.hpp"

  /**
 * @brief Predict the price of an option using the Monte Carlo method.
 * @details This function predicts the price of an option using the Monte Carlo method.
 * @param points The number of points to use in the Monte Carlo method.
 * @param function The function to use in the Monte Carlo method.
 * @param assetPtrs The vector of pointers to the Asset objects.
 * @param variance The variance of the Monte Carlo method.
 * @param coefficients The coefficients of the function.
 * @param strike_price The strike price of the option.
 * @param predicted_assets_prices The vector that will contain the predicted assets prices.
 * @return A pair containing the price of the option and the computation time in microseconds.
 */
std::pair<double, double> monteCarloPricePrediction(size_t points,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double &variance,
                                                    const double strike_price,
                                                    std::vector<double> &predicted_assets_prices,
                                                    const OptionType &option_type,
                                                    MonteCarloError &error);

  /**
 * @brief Generate a random point for the Monte Carlo simulation.
 * @details This function generates a random point for the Monte Carlo simulation.
 * @param random_point1 Vector to store the first random point.
 * @param random_point2 Vector to store the second random point.
 * @param assetPtrs Vector of pointers to the Asset objects.
 * @param predicted_assets_prices Vector to store the predicted asset prices.
 */
void generateRandomPoint(std::vector<double> &random_point1,
                         std::vector<double> &random_point2,
                         const std::vector<const Asset *> &assetPtrs,
                         std::vector<double> &predicted_assets_prices,
                         const OptionType &option_type,
                         const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<double>> &zeta_matrix,
                         const uint num_days_to_simulate);


#endif
