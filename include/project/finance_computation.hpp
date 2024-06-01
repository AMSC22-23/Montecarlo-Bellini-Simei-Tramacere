#ifndef FINANCE_COMPUTATION_HPP
    #define FINANCE_COMPUTATION_HPP

  /**
 * @brief This function is the core of the finance project: 
 * it embeds multiple methods that are used to compute
 * the option price using the Monte Carlo method
 * @details The function loads the assets from the CSV files,
 * calculates the strike price, creates the payoff function,
 * and computes the option price using the Monte Carlo method.
 */
void financeComputation();

#endif