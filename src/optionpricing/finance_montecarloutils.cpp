#include "../../include/optionpricing/finance_montecarloutils.hpp"

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

  // Function to generate a metrix of random numbers from a normal distribution
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

  // Function to compute a vector-vector multiplication
double VVMult(const std::vector<std::vector<double>> &matrix, size_t rowIdx, const std::vector<double> &vector)
{
    double result = 0.0;
    for (size_t i = 0; i < vector.size(); ++i)
    {
        result += matrix[rowIdx][i] * vector[i];
    }
    return result;
}

  // Function that shifts the seed
uint32_t xorshift(uint32_t seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}
