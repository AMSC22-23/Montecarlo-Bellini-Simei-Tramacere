/**
 * @file finance_pricingutils.hpp
 * @brief This file contains utility functions for option pricing in finance.
 */

#ifndef FINANCE_PRICINGUTILS_HPP
    #define FINANCE_PRICINGUTILS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "finance_inputmanager.hpp"
#include "asset.hpp"
#include "finance_montecarlo.hpp"
#include "optionparameters.hpp"
#include "finance_enums.hpp"

/**
 * @brief Calculates the value of the standard normal distribution function.
 * @param x The input value.
 * @return The value of the standard normal distribution function at x.
 */
double phi(const double x);

/**
 * @brief Writes the results of the option pricing to a file.
 * @param assets A vector of assets used in the computation.
 * @param result A pair of doubles containing the option price and computation time.
 * @param standard_error The standard error of the option price estimate.
 * @param function The function used in the computation.
 * @param num_simulations The number of Monte Carlo simulations performed.
 */
void writeResultsToFile(const std::vector<Asset> &assets,
                        const std::pair<double, double> &result,
                        const double &standard_error,
                        const std::string &function,
                        const size_t &num_simulations,
                        const OptionType &option_type);

/**
 * @brief Computes the option price using the Black-Scholes model.
 * @param assetPtrs A vector of pointers to assets.
 * @param strike_price The strike price of the option.
 * @return The computed Black-Scholes option price.
 */
double computeBlackScholesOptionPrice(const std::vector<const Asset *> &assetPtrs,
                                      const double &strike_price);

/**
 * @brief Prompts the user to select the type of option.
 * @return The selected option type.
 */
OptionType getOptionTypeFromUser();

/**
 * @brief Prompts the user to select the type of asset count.
 * @return The selected asset count type.
 */
AssetCountType getAssetCountTypeFromUser();

#endif