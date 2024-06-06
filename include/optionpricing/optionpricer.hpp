  /**
 * @file optionpricer.hpp
 * @brief This file contains the declarationof the financeComputation function.
 */

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
#include "finance_pricingutils.hpp"

  /**
 * @brief This function is the core of the finance project: 
 * it embeds multiple methods that are used to compute
 * the option price using the Monte Carlo method.
 * @details The function loads the assets from the CSV files,
 * calculates the strike price, creates the payoff function,
 * and computes the option price using the Monte Carlo method.
 */
void financeComputation();

#endif
