#include "../../include/optionpricing/finance_montecarlo.hpp"

  // Function to calculate the option price prediction using the Monte Carlo method
  // The function is the core of the finance oriented project, which is used to predict
  // the option price prediction using the Monte Carlo method.
std::pair<double, double> monteCarloPricePrediction(size_t points,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double &variance,
                                                    const double strike_price,
                                                    std::vector<double> &predicted_assets_prices,
                                                    const OptionType &option_type,
                                                    MonteCarloError &error)
{
    double C                   = 0.0;
    double C0                  = 0.0;
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result1             = 0.0;
    double result2             = 0.0;
    double r                   = 0.05;
    double T                   = 1.0;
      // Number of days to simulate (1 day for European option, 252 days for Asian option
    uint num_days_to_simulate = 1;
    if (option_type == OptionType::Asian)
    {
        num_days_to_simulate = 252;
    }

      // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

      // Calculate the covariance matrix
    CovarianceError cov_error;
    std::vector<std::vector<double>> covariance_matrix = calculateCovarianceMatrix(assetPtrs, cov_error);

      // Check if the covariance matrix was calculated successfully
    if (cov_error != CovarianceError::Success)
    {
        std::cerr << "Error calculating the covariance matrix" << std::endl;
        return std::make_pair(0.0, 0.0);
    }

      // Calculate the Cholesky factorization of the covariance matrix
    std::vector<std::vector<double>> A = choleskyFactorization(covariance_matrix, 1.0);

      // Check if the matrix is positive-definite
    if (A.empty())
    {
        std::cerr << "Matrix is not positive-definite" << std::endl;
        return std::make_pair(0.0, 0.0);
    }

    // Zeta matrix
    std::vector<std::vector<double>> zeta_matrix(num_days_to_simulate, std::vector<double>(assetPtrs.size(), 0.0));
    fillZetaMatrix(zeta_matrix);

#pragma omp parallel
    {
        // Random point vectors
        std::vector<double> random_point_vector1(assetPtrs.size(), 0.0);
        std::vector<double> random_point_vector2(assetPtrs.size(), 0.0);

#pragma omp for reduction(+ : total_value, total_squared_value) schedule(dynamic)
        for (size_t i = 0; i < points / 2; ++i)
        {
              // Generate random point
              generateRandomPoint(random_point_vector1,
                                  random_point_vector2,
                                  assetPtrs,
                                  predicted_assets_prices,
                                  option_type,
                                  A,
                                  zeta_matrix,
                                  num_days_to_simulate);

              // Check if the random point vector is not empty
            if (random_point_vector1.size() != 0 && random_point_vector2.size() != 0)
            {
                error   = MonteCarloError::Success;
                result1 = 0.0;
                result2 = 0.0;

                  // Evaluate the payoff function with the random point
                for (size_t i = 0; i < random_point_vector1.size(); ++i)
                {
                    result1 += random_point_vector1[i];
                    result2 += random_point_vector2[i];
                }

                result1 = std::max(0.0, (result1 - strike_price));
                result2 = std::max(0.0, (result2 - strike_price));

                total_value += result1 + result2;
                total_squared_value += result1 * result1 +
                                       result2 * result2;
            }
            else
            {
                error = MonteCarloError::PointGenerationFailed;
                i     = points / 2;
            }
        }
    }

    if (error == MonteCarloError::PointGenerationFailed)
    {
        return std::make_pair(0.0, 0.0);
    }

      // Calculate the option price
    C  = total_value / static_cast<double>(points);
    C0 = C * exp(-r * T);

      // Calculate the variance
    variance = total_squared_value / static_cast<double>(points) - (total_value / static_cast<double>(points)) * (total_value / static_cast<double>(points));

      // Stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(C0, static_cast<double>(duration.count()));
}

  // Function to generate a random point
void generateRandomPoint(std::vector<double> &random_point1,
                         std::vector<double> &random_point2,
                         const std::vector<const Asset *> &assetPtrs,
                         std::vector<double> &predicted_assets_prices,
                         const OptionType &option_type,
                         const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<double>> &zeta_matrix,
                         const uint num_days_to_simulate)
{
    double T = 1.0;                       /**Time to maturity */
    double r = 0.05;                      /**Risk-free rate */
    double dt = T / num_days_to_simulate; /**Time step */

    try
    {
        // thread_local std::mt19937 eng(xorshift(seed));

        for (size_t i = 0; i < assetPtrs.size(); ++i)
        {
              // Geometry Brownian Motion price:
            std::normal_distribution<double> distribution(0, 1);
            double prices1[num_days_to_simulate + 1];
            double prices2[num_days_to_simulate + 1];

              // Asian option prices
            double asian_prices1 = 0.0;
            double asian_prices2 = 0.0;

              // Initial price
            prices1[0] = assetPtrs[i]->getLastRealValue();
            prices2[0] = assetPtrs[i]->getLastRealValue();

            for (uint step = 1; step < num_days_to_simulate + 1; ++step)
            {
                  // Perform the dot product to get the random number
                double num = VVMult(A, i, zeta_matrix[step - 1]);

                  // Calculate the price
                prices1[step] = prices1[step - 1] * exp((r -
                                                         0.5 *
                                                             assetPtrs[i]->getReturnStdDev() *
                                                             assetPtrs[i]->getReturnStdDev()) *
                                                            dt +
                                                        sqrt(dt) *
                                                            num);
                prices2[step] = prices2[step - 1] * exp((r -
                                                         0.5 *
                                                             assetPtrs[i]->getReturnStdDev() *
                                                             assetPtrs[i]->getReturnStdDev()) *
                                                            dt -
                                                        sqrt(dt) *
                                                            num);
                  // Calculate the Asian option prices
                if (option_type == OptionType::Asian)
                {
                    asian_prices1 += prices1[step];
                    asian_prices2 += prices2[step];
                }
            }
              // Calculate the random point
            if (option_type == OptionType::Asian)
            {
                random_point1[i] = asian_prices1 / num_days_to_simulate;
                random_point2[i] = asian_prices2 / num_days_to_simulate;
            }
            else if (option_type == OptionType::European)
            {
                random_point1[i] = prices1[num_days_to_simulate];
                random_point2[i] = prices2[num_days_to_simulate];
            }

#pragma omp critical
            {

                  // Calculate the predicted asset prices
                predicted_assets_prices[i] += prices1[num_days_to_simulate] + prices2[num_days_to_simulate];
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        random_point1.clear();
        random_point2.clear();
        return;
    }
}