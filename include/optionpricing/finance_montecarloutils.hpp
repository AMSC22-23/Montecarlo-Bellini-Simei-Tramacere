/**
 * @file finance_montecarloutils.hpp
 * @brief This file contains declarations related to Monte Carlo simulation utilities for pricing options.
 */

#ifndef PROJECT_FINANCEMONTECARLOUTILS_HPP
    #define PROJECT_FINANCEMONTECARLOUTILS_HPP

#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <random>

#include "asset.hpp"
#include "finance_enums.hpp"
#include "finance_inputmanager.hpp"
#include "../integration/geometry/hyperrectangle.hpp"


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
double calculateCovariance(const Asset &asset1, const Asset &asset2, CovarianceError &error);

  /**
 * @brief Calculate the covariance matrix for a set of assets.
 * @details This function calculates the covariance matrix for a set of assets.
 * @param assetPtrs Vector of pointers to the Asset objects.
 * @return The covariance matrix.
 */
std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<const Asset *> &assetPtrs, CovarianceError &error);

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