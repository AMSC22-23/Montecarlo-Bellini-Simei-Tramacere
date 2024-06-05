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

  /**
 * @brief Predict the price of an option using the Monte Carlo method.
 * @details This function predicts the price of an option using the Monte Carlo method.
 * @param points The number of points to use in the Monte Carlo method.
 * @param function The function to use in the Monte Carlo method.
 * @param hyperrectangle The hyperrectangle that contains the integration bounds.
 * @param assetPtrs The vector of pointers to the Asset objects.
 * @param std_dev_from_mean The standard deviation from the mean.
 * @param variance The variance of the Monte Carlo method.
 * @param coefficients The coefficients of the function.
 * @param strike_price The strike price of the option.
 * @param predicted_assets_prices The vector that will contain the predicted assets prices.
 * @return A pair containing the price of the option and the computation time in microseconds.
 */
std::pair<double, double> monteCarloPricePrediction(size_t points,
                                                    const std::string &function,
                                                    HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double std_dev_from_mean,
                                                    double &variance,
                                                    std::vector<double> coefficients,
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
 * @param std_dev_from_mean Standard deviation from the mean.
 * @param predicted_assets_prices Vector to store the predicted asset prices.
 */
void generateRandomPoint(std::vector<double> &random_point1,
                         std::vector<double> &random_point2,
                         const std::vector<const Asset *> &assetPtrs,
                         const double std_dev_from_mean,
                         std::vector<double> &predicted_assets_prices,
                         const OptionType &option_type);

  /**
 * @brief XOR shift random number generator.
 * @details This function generates a random number using the XOR shift algorithm.
 * @param seed The seed for the random number generator.
 * @return A random number.
 */
uint32_t xorshift(uint32_t seed);

  /**
 * @brief Calculate the covariance between two assets.
 * @details This function calculates the covariance between the daily returns of two assets.
 * @param asset1 The first asset.
 * @param asset2 The second asset.
 * @return The covariance between the two assets.
 */
double calculateCovariance(const Asset &asset1, const Asset &asset2);

  /**
 * @brief Calculate the covariance matrix for a set of assets.
 * @details This function calculates the covariance matrix for a set of assets.
 * @param assetPtrs Vector of pointers to the Asset objects.
 * @return The covariance matrix.
 */
std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<const Asset *> &assetPtrs);

  /**
 * @brief Perform Cholesky factorization on a matrix.
 * @details This function performs Cholesky factorization on a matrix.
 * @param A The matrix to factorize.
 * @param step_size The step size for the factorization.
 * @return The lower triangular matrix resulting from the factorization.
 */
std::vector<std::vector<double>> choleskyFactorization(const std::vector<std::vector<double>> &A, double step_size);

  /**
 * @brief Fill a matrix with random values from a normal distribution.
 * @details This function fills a matrix with random values from a normal distribution.
 * @param zeta_matrix The matrix to fill.
 */
void fillZetaMatrix(std::vector<std::vector<double>> &zeta_matrix);

  /**
 * @brief Perform vector-vector multiplication.
 * @details This function performs vector-vector multiplication for a specific row of the matrix.
 * @param matrix The matrix.
 * @param rowIdx The index of the row to use.
 * @param vector The vector to multiply.
 * @return The result of the multiplication.
 */
double VVMult(const std::vector<std::vector<double>> &matrix, size_t rowIdx, const std::vector<double> &vector);

#endif
