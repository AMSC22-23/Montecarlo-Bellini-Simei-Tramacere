#ifndef PROJECT_FINANCEMONTECARLO_HPP
    #define PROJECT_FINANCEMONTECARLO_HPP

#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <random>

#include "hyperrectangle.hpp"
#include "asset.hpp"

/**
 * @brief This function is used to predict the price of an option using the Monte Carlo method
 * @details The function predicts the price of an option using the Monte Carlo method
 * @param points The number of points to use in the Monte Carlo method
 * @param function The function to use in the Monte Carlo method
 * @param hyperrectangle The hyperrectangle that contains the integration bounds
 * @param assetPtrs The vector of pointers to the Asset objects
 * @param std_dev_from_mean The standard deviation from the mean
 * @param variance The variance of the Monte Carlo method
 * @param coefficients The coefficients of the function
 * @param strike_price The strike price of the option
 * @param predicted_assets_prices The vector that will contain the predicted assets prices
 * @return A pair containing the price of the option and the standard deviation
 */
std::pair<double, double> montecarloPricePrediction(size_t points,
                                                    const std::string &function,
                                                    HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double std_dev_from_mean,
                                                    double &variance,
                                                    std::vector<double> coefficients,
                                                    const double strike_price,
                                                    std::vector<double> &predicted_assets_prices);


void generateRandomPoint(std::vector<double> &random_point1, std::vector<double> &random_point2, 
                                                const std::vector<const Asset *> &assetPtrs,
                                                const double std_dev_from_mean,
                                                std::vector<double> &predicted_assets_prices);

uint32_t xorshift(uint32_t seed);

#endif