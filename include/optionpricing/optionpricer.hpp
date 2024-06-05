#ifndef OPTION_PRICER_HPP
    #define OPTION_PRICER_HPP

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
 * @brief This function is the core of the finance project: 
 * it embeds multiple methods that are used to compute
 * the option price using the Monte Carlo method.
 * @details The function loads the assets from the CSV files,
 * calculates the strike price, creates the payoff function,
 * and computes the option price using the Monte Carlo method.
 */
void financeComputation();

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
                        const size_t &num_simulations);

  /**
 * @brief Computes the option price using the Black-Scholes model.
 * @param assetPtrs A vector of pointers to assets.
 * @param strike_price The strike price of the option.
 * @return The computed Black-Scholes option price.
 */
double computeBlackScholesOptionPrice(const std::vector<const Asset *> &assetPtrs,
                                      const double &strike_price);

#endif
