#include "../../include/optionpricing/finance_montecarlo.hpp"

  // Calculate covariance between two assets
double calculateCovariance(const Asset &asset1, const Asset &asset2, CovarianceError &error)
{
    double covariance = 0.0;
    double mean1      = asset1.getReturnMean();
    double mean2      = asset2.getReturnMean();

    try
    {
          // Check if the sizes of daily returns match
        if (asset1.getDailyReturnsSize() != asset2.getDailyReturnsSize())
        {
            error = CovarianceError::Failure;
            return covariance;
        }

          // Calculate covariance
        size_t dataSize = asset1.getDailyReturnsSize();
        for (size_t i = 0; i < dataSize; ++i)
        {
            covariance += (asset1.getDailyReturn(i) - mean1) * (asset2.getDailyReturn(i) - mean2);
        }
        covariance /= (dataSize - 1);
        error       = CovarianceError::Success;
    }
    catch (const std::exception &e)
    {
        error = CovarianceError::Failure;
    }

    return covariance;
}

  // Calculate covariance matrix for a vector of assets
std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<const Asset *> &assetPtrs, CovarianceError &error)
{
    size_t numAssets = assetPtrs.size();
    std::vector<std::vector<double>> covarianceMatrix(numAssets, std::vector<double>(numAssets, 0.0));

    try
    {
          // Fill covariance matrix
        for (size_t i = 0; i < numAssets; ++i)
        {
            for (size_t j = 0; j < numAssets; ++j)
            {
                covarianceMatrix[i][j] = calculateCovariance(*assetPtrs[i], *assetPtrs[j], error);
                if (error != CovarianceError::Success)
                {
                    return covarianceMatrix;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        error = CovarianceError::Failure;
    }

    return covarianceMatrix;
}

std::vector<std::vector<double>> choleskyFactorization(const std::vector<std::vector<double>> &A, double step_size)
{
    int n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (int c = 0; c < n; ++c)
    {
        double sum = 0.0;

          // Efficiently calculate L(c, c) using previously computed L elements
        for (int k = 0; k < c; ++k)
        {
            sum += L[c][k] * L[c][k];
        }
        L[c][c] = sqrt(A[c][c] - sum);  // Handle potential negative values by returning an empty matrix

          // Check for positive-definite condition
        if (L[c][c] <= 0.0)
        {
            return std::vector<std::vector<double>>();  // Matrix not positive-definite
        }

          // Update the rest of the c-th column of L
        if (c + step_size < n)
        {
            for (int i = c + 1; i < n; ++i)
            {
                sum = 0.0;
                for (int k = 0; k < c; ++k)
                {
                    sum += L[i][k] * L[c][k];
                }
                L[i][c] = (A[i][c] - sum) / L[c][c];
            }
        }
    }

    return L;
}

void fillZetaMatrix(std::vector<std::vector<double>> &zeta_matrix)
{
    std::random_device rd;
    std::mt19937 eng(rd());
    for (size_t i = 0; i < zeta_matrix.size(); ++i)
    {
        std::normal_distribution<double> distribution(0, 1);
        for (size_t j = 0; j < zeta_matrix[i].size(); ++j)
        {
            zeta_matrix[i][j] = distribution(eng);
        }
    }
}

double VVMult(const std::vector<std::vector<double>> &matrix, size_t rowIdx, const std::vector<double> &vector)
{
    double result = 0.0;
    for (size_t i = 0; i < vector.size(); ++i)
    {
        result += matrix[rowIdx][i] * vector[i];
    }
    return result;
}

uint32_t xorshift(uint32_t seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

  // Function to calculate the option price prediction using the Monte Carlo method
  // The function is the core of the finance oriented project, which is used to predict
  // the option price prediction using the Monte Carlo method.
std::pair<double, double> monteCarloPricePrediction(size_t points,
                                                    const std::string &function,
                                                    HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    const double std_dev_from_mean,
                                                    double &variance,
                                                    std::vector<double> coefficients,
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
    auto   start               = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        std::vector<double> random_point_vector1(assetPtrs.size(), 0.0);
        std::vector<double> random_point_vector2(assetPtrs.size(), 0.0);

#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < points / 2; ++i)
        {

            generateRandomPoint(random_point_vector1, random_point_vector2, assetPtrs, std_dev_from_mean, predicted_assets_prices, option_type);

              // Check if the random point vector is not empty
            if (random_point_vector1.size() != 0 && random_point_vector2.size() != 0)
            {
                error = MonteCarloError::Success;
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

                total_value         += result1 + result2;
                total_squared_value += result1 * result1 +
                                       result2 * result2;
            }
            else
            {
                error = MonteCarloError::PointGenerationFailed;
                i = points / 2;
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
                         const double std_dev_from_mean,
                         std::vector<double> &predicted_assets_prices,
                         const OptionType &option_type)
{
    uint     num_days_to_simulate = 252;
    double   T                    = 1.0;
    double   r                    = 0.05;
    double   dt                   = T / num_days_to_simulate;
    uint32_t seed                 = 123456789;
    CovarianceError error;
    std::vector<std::vector<double>> covariance_matrix = calculateCovarianceMatrix(assetPtrs, error);

    if (error != CovarianceError::Success)
    {
        std::cerr << "Error calculating the covariance matrix" << std::endl;
        random_point1.clear();
        random_point2.clear();
        return;
    }

    std::vector<std::vector<double>> A = choleskyFactorization(covariance_matrix, 1.0);

    if (A.empty())
    {
        std::cerr << "Matrix is not positive-definite" << std::endl;
        random_point1.clear();
        random_point2.clear();
        return;
    }

    std::vector<std::vector<double>> zeta_matrix(num_days_to_simulate, std::vector<double>(assetPtrs.size(), 0.0));
    fillZetaMatrix(zeta_matrix);

    try
    {
        thread_local std::mt19937 eng(xorshift(seed));

#pragma omp parallel for
        for (size_t i = 0; i < assetPtrs.size(); ++i)
        {
              // Geometry Brownian Motion price:
            std::normal_distribution<double> distribution(0, 1);
            double prices1[num_days_to_simulate + 1];
            double prices2[num_days_to_simulate + 1];

            double asian_prices1 = 0.0;
            double asian_prices2 = 0.0;
            prices1[0]           = assetPtrs[i]->getLastRealValue();
            prices2[0]           = assetPtrs[i]->getLastRealValue();

            for (uint step = 1; step < num_days_to_simulate + 1; ++step)
            {
                double num = VVMult(A, i, zeta_matrix[step - 1]);

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
                if (option_type == OptionType::Asian && step % 21 == 0)
                {
                    asian_prices1 += prices1[step];
                    asian_prices2 += prices2[step];
                }
            }

#pragma omp critical
            {
                if (option_type == OptionType::Asian)
                {
                    random_point1[i] = asian_prices1 / 12.0;
                    random_point2[i] = asian_prices2 / 12.0;
                }
                else if (option_type == OptionType::European)
                {
                    random_point1[i] = prices1[num_days_to_simulate];
                    random_point2[i] = prices2[num_days_to_simulate];
                }
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