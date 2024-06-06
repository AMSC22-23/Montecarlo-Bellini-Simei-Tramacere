#include <cuda_runtime.h>
#include <math.h>

extern "C++"
{
#include "../include/optionpricing/asset.hpp"
#include "../include/optionpricing/optionpricer.hpp"
#include "../include/optionpricing/finance_montecarlo.hpp"
#include "../include/optionpricing/optionparameters.hpp"
#include "../include/optionpricing/finance_inputmanager.hpp"
}

float cuda_calculateCovariance(const Asset &asset1, const Asset &asset2, CovarianceError &error)
{
    float covariance = 0.0;
    float mean1 = asset1.getReturnMean();
    float mean2 = asset2.getReturnMean();

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
        error = CovarianceError::Success;
    }
    catch (const std::exception &e)
    {
        error = CovarianceError::Failure;
    }

    return covariance;
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

float cuda_phi(float x)
{
    static const float RT2PI = sqrt(4.0 * acos(0.0));

    static const float SPLIT = 7.07106781186547;

    static const float N0 = 220.206867912376;
    static const float N1 = 221.213596169931;
    static const float N2 = 112.079291497871;
    static const float N3 = 33.912866078383;
    static const float N4 = 6.37396220353165;
    static const float N5 = 0.700383064443688;
    static const float N6 = 3.52624965998911e-02;
    static const float M0 = 440.413735824752;
    static const float M1 = 793.826512519948;
    static const float M2 = 637.333633378831;
    static const float M3 = 296.564248779674;
    static const float M4 = 86.7807322029461;
    static const float M5 = 16.064177579207;
    static const float M6 = 1.75566716318264;
    static const float M7 = 8.83883476483184e-02;

    const float z = fabs(x);
    float c = 0.0;

    if (z <= 37.0)
    {
        const float e = exp(-z * z / 2.0);
        if (z < SPLIT)
        {
            const float n = (((((N6 * z + N5) * z + N4) * z + N3) * z + N2) * z + N1) * z + N0;
            const float d = ((((((M7 * z + M6) * z + M5) * z + M4) * z + M3) * z + M2) * z + M1) * z + M0;
            c = e * n / d;
        }
        else
        {
            const float f = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))));
            c = e / (RT2PI * f);
        }
    }
    if (x <= 0.0)
        return c;
    else
        return 1 - c;
}

__global__ void priceEuropeanOption(float *total_payoff, float *total_squared_value,
                                    const float *assets_returns, const float *assets_std_devs,
                                    long long int n, float *assets_closing_values, int strike_price,
                                    float *predicted_assets_prices, int num_assets,
                                    float *matrix, float *vector, int num_days_to_simulate)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
    {

        float payoff_function1 = 0.0;
        float payoff_function2 = 0.0;
        float dt = 1.0 / num_days_to_simulate;
        float prices1[253];
        float prices2[253];
        float r = 0.05;
        float Z = 0.0;

        for (size_t asset_idx = 0; asset_idx < num_assets; asset_idx++)
        {

            prices1[0] = assets_closing_values[asset_idx];
            prices2[0] = assets_closing_values[asset_idx];
            // printf( "prices1[0] : %f prices2[0] : %f \n", prices1[0], prices2[0]);

            for (uint step = 1; step < num_days_to_simulate + 1; ++step)
            {
                Z = 0.0;
                for (size_t i = 0; i < num_assets; ++i)
                {
                    Z += matrix[asset_idx * num_assets + i] * vector[(step - 1) * num_assets + i];
                }

                prices1[step] = prices1[step - 1] * exp((r - 0.5 * assets_std_devs[asset_idx] *
                                                                 assets_std_devs[asset_idx]) *
                                                            dt -
                                                        assets_std_devs[asset_idx] * sqrt(dt) * Z);

                prices2[step] = prices2[step - 1] * exp((r - 0.5 * assets_std_devs[asset_idx] *
                                                                 assets_std_devs[asset_idx]) *
                                                            dt +
                                                        assets_std_devs[asset_idx] * sqrt(dt) * Z);
            }

            payoff_function1 += prices1[num_days_to_simulate];
            payoff_function2 += prices2[num_days_to_simulate];

            atomicAdd(&predicted_assets_prices[asset_idx], (prices1[num_days_to_simulate] + prices2[num_days_to_simulate]) / 2);
            atomicAdd(&total_squared_value[asset_idx],
                      (prices1[num_days_to_simulate] * prices1[num_days_to_simulate] + prices2[num_days_to_simulate] * prices2[num_days_to_simulate])/2);
        }

        if (payoff_function1 > strike_price)
            payoff_function1 -= strike_price;
        else
            payoff_function1 = 0.0;

        if (payoff_function2 > strike_price)
            payoff_function2 -= strike_price;
        else
            payoff_function2 = 0.0;

        atomicAdd(&total_payoff[tid % 1000000], (payoff_function1 + payoff_function2) / 2);
    }
}

__global__ void priceAsianOption(float *total_payoff, float *total_squared_value,
                                 const float *assets_returns, const float *assets_std_devs,
                                 long long int n, float *assets_closing_values, int strike_price,
                                 float *predicted_assets_prices, int num_assets,
                                 float *matrix, float *vector, int num_days_to_simulate)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
    {

        float payoff_function1 = 0.0;
        float payoff_function2 = 0.0;
        float mean1 = 0.0;
        float mean2 = 0.0;
        float dt = 1.0 / num_days_to_simulate;
        float prices1[253];
        float prices2[253];
        float r = 0.05;
        float Z = 0.0;

        for (size_t asset_idx = 0; asset_idx < num_assets; asset_idx++)
        {

            prices1[0] = assets_closing_values[asset_idx];
            prices2[0] = assets_closing_values[asset_idx];

            for (uint step = 1; step < num_days_to_simulate + 1; ++step)
            {
                Z = 0.0;
                for (size_t i = 0; i < num_assets; ++i)
                {
                    Z += matrix[asset_idx * num_assets + i] * vector[(step - 1) * num_assets + i];
                }

                prices1[step] = prices1[step - 1] * exp((r - 0.5 * assets_std_devs[asset_idx] *
                                                                 assets_std_devs[asset_idx]) *
                                                            dt -
                                                        assets_std_devs[asset_idx] * sqrt(dt) * Z);

                prices2[step] = prices2[step - 1] * exp((r - 0.5 * assets_std_devs[asset_idx] *
                                                                 assets_std_devs[asset_idx]) *
                                                            dt +
                                                        assets_std_devs[asset_idx] * sqrt(dt) * Z);
                mean1 += prices1[step];
                mean2 += prices2[step];
                //printf("prices1[%d] : %f prices2[%d] : %f \n", step, prices1[num_days_to_simulate], step, prices2[num_days_to_simulate]);
            }
            mean1 /= num_days_to_simulate;
            mean2 /= num_days_to_simulate;
            payoff_function1 += mean1;
            payoff_function2 += mean2;

            atomicAdd(&predicted_assets_prices[asset_idx], (prices1[num_days_to_simulate] + prices2[num_days_to_simulate]) / 2);
            atomicAdd(&total_squared_value[asset_idx],
                      (prices1[num_days_to_simulate] * prices1[num_days_to_simulate] + prices2[num_days_to_simulate] * prices2[num_days_to_simulate]) / 2);
        }

        if (payoff_function1 > strike_price)
            payoff_function1 -= strike_price;
        else
            payoff_function1 = 0.0;

        if (payoff_function2 > strike_price)
            payoff_function2 -= strike_price;
        else
            payoff_function2 = 0.0;

        atomicAdd(&total_payoff[tid % 1000000], (payoff_function1 + payoff_function2) / 2);
    }
}

__global__ void printFunction(long long int n, char *function, const double *coefficients, int number_of_coefficients)
{
    printf("Function: %s\n", function);
    printf("n: %ld\n", n);
    for (size_t i = 0; i < number_of_coefficients; ++i)
    {
        printf("Coefficient[%ld]: %f\n", i, coefficients[i]);
    }
    printf("Number of coefficients: %d\n", number_of_coefficients);
}

extern std::pair<double, double> kernel_wrapper(long long int n, const std::string &function,
                                                const std::vector<const Asset *> &assetPtrs, double *variance,
                                                std::vector<double> coefficients, double strike_price, OptionType option_type)
{
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads_per_block = 256;
    dim3 number_of_blocks = (n + threads_per_block.x - 1) / threads_per_block.x;

    uint num_assets = assetPtrs.size();
    uint num_days_to_simulate = 1;

    double S = 0.0;
    double r = 0.05;
    double sigma = 0.0;
    double T = 1;
    double BS_option_price = 0.0;

    // Create and copy function and coefficients to device
    char *d_function;
    size_t function_size = function.size() + 1; // Include the null terminator
    gpuErrchk(cudaMalloc((void **)&d_function, function_size * sizeof(char)));
    gpuErrchk(cudaMemcpy(d_function, function.c_str(), function_size * sizeof(char), cudaMemcpyHostToDevice));

    double *d_coefficients;
    gpuErrchk(cudaMalloc((void **)&d_coefficients, coefficients.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_coefficients, coefficients.data(), coefficients.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Call the CUDA kernel to print the function and coefficients
    printFunction<<<1, 1>>>(n, d_function, d_coefficients, coefficients.size());

    CovarianceError cov_error;
    cov_error = CovarianceError::Success;

    size_t numAssets = assetPtrs.size();
    float covarianceMatrix[num_assets][num_assets];

    try
    {
        // Fill covariance matrix
        for (size_t i = 0; i < numAssets; ++i)
        {
            for (size_t j = 0; j < numAssets; ++j)
            {
                covarianceMatrix[i][j] = cuda_calculateCovariance(*assetPtrs[i], *assetPtrs[j], cov_error);
            }
        }
    }
    catch (const std::exception &e)
    {
        cov_error = CovarianceError::Failure;
    }

    if (cov_error != CovarianceError::Success)
    {
        std::cerr << "Error calculating the covariance matrix" << std::endl;
        return std::make_pair(0.0, 0.0);
    }

    float A[num_assets][num_assets];

    for (uint c = 0; c < num_assets; ++c)
    {
        float sum = 0.0;

        // Efficiently calculate A(c, c) using previously computed A elements
        for (uint k = 0; k < c; ++k)
        {
            sum += A[c][k] * A[c][k];
        }
        A[c][c] = sqrt(covarianceMatrix[c][c] - sum); // Handle potential negative values by returning an empty matrix
        if (isnan(A[c][c]))
        {
            printf("Matrix not positive-definite\n");
            return std::make_pair(0.0, 0.0); // Matrix not positive-definite
        }
        // Check for positive-definite condition
        if (A[c][c] <= 0.0)
        {
            printf("Matrix not positive-definite\n");
            return std::make_pair(0.0, 0.0); // Matrix not positive-definite
        }

        // Update the rest of the c-th column of A
        if (c + 1.0 < num_assets)
        {
            for (uint i = c + 1; i < num_assets; ++i)
            {
                sum = 0.0;
                for (uint k = 0; k < c; ++k)
                {
                    sum += A[i][k] * A[c][k];
                }
                A[i][c] = (covarianceMatrix[i][c] - sum) / A[c][c];
            }
        }
    }

    float zeta_matrix[num_days_to_simulate][num_assets];
    std::random_device rd;
    std::mt19937 eng(rd());
    for (size_t i = 0; i < num_days_to_simulate; ++i)
    {
        std::normal_distribution<float> distribution(0, 1);
        for (size_t j = 0; j < num_assets; ++j)
        {
            zeta_matrix[i][j] = distribution(eng);
        }
    }

    float *d_zeta_matrix;
    gpuErrchk(cudaMalloc((void **)&d_zeta_matrix, num_days_to_simulate * num_assets * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_zeta_matrix, zeta_matrix, num_days_to_simulate * num_assets * sizeof(float), cudaMemcpyHostToDevice));

    float *d_A;
    gpuErrchk(cudaMalloc((void **)&d_A, num_assets * num_assets * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_A, A, num_assets * num_assets * sizeof(float), cudaMemcpyHostToDevice));

    // Save the assets main data
    float *d_assets_returns;
    float *d_assets_std_devs;
    float *d_assets_last_values;
    gpuErrchk(cudaMalloc((void **)&d_assets_returns, num_assets * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_assets_std_devs, num_assets * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_assets_last_values, num_assets * sizeof(float)));

    for (size_t i = 0; i < num_assets; i++)
    {
        float return_mean = static_cast<float>(assetPtrs[i]->getReturnMean());
        gpuErrchk(cudaMemcpy(&d_assets_returns[i], &return_mean, sizeof(float), cudaMemcpyHostToDevice));

        float return_std_dev = static_cast<float>(assetPtrs[i]->getReturnStdDev());
        gpuErrchk(cudaMemcpy(&d_assets_std_devs[i], &return_std_dev, sizeof(float), cudaMemcpyHostToDevice));

        float last_value = static_cast<float>(assetPtrs[i]->getLastRealValue());
        gpuErrchk(cudaMemcpy(&d_assets_last_values[i], &last_value, sizeof(float), cudaMemcpyHostToDevice));
    }

    float *d_total_squared_value, *d_total_payoff;
    gpuErrchk(cudaMalloc(&d_total_squared_value, num_assets * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_total_payoff, 1000000 * sizeof(float)));

    float predicted_assets_prices[num_assets];
    float *d_predicted_assets_prices;
    gpuErrchk(cudaMalloc(&d_predicted_assets_prices, num_assets * sizeof(float)));

    if (option_type == OptionType::European)
    {
        num_days_to_simulate = 1;
        priceEuropeanOption<<<number_of_blocks, threads_per_block>>>(d_total_payoff, d_total_squared_value,
                                                                     d_assets_returns, d_assets_std_devs, n,
                                                                     d_assets_last_values, strike_price,
                                                                     d_predicted_assets_prices, num_assets,
                                                                     d_A, d_zeta_matrix, num_days_to_simulate);
    }
    else
    {
        num_days_to_simulate = 12;
        priceAsianOption<<<number_of_blocks, threads_per_block>>>(d_total_payoff, d_total_squared_value,
                                                                  d_assets_returns, d_assets_std_devs, n,
                                                                  d_assets_last_values, strike_price,
                                                                  d_predicted_assets_prices, num_assets,
                                                                  d_A, d_zeta_matrix, num_days_to_simulate);
    }
    cudaDeviceSynchronize();

    float host_total_squared_value[num_assets];
    gpuErrchk(cudaMemcpy(host_total_squared_value, d_total_squared_value, num_assets * sizeof(float), cudaMemcpyDeviceToHost));

    float host_assets_prices[num_assets];
    gpuErrchk(cudaMemcpy(host_assets_prices, d_predicted_assets_prices, num_assets * sizeof(float), cudaMemcpyDeviceToHost));

    float total_payoff[1000000];
    gpuErrchk(cudaMemcpy(&total_payoff, d_total_payoff, 1000000 * sizeof(float), cudaMemcpyDeviceToHost));

    float total_squared_value = 0.0;

    for (size_t i = 0; i < num_assets; i++)
    {
        total_squared_value += (host_total_squared_value[i]);
        predicted_assets_prices[i] = (host_assets_prices[i] / n);
        std::cout << "The predicted future price (30 days) of one " << assetPtrs[i]->getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }

    float option_payoff = 0.0;
    for (size_t i = 0; i < 1000000; i++)
        option_payoff += total_payoff[i] / n;

    option_payoff = option_payoff * exp(-r * T);

    // calculate the variance
    *variance = total_squared_value / n - (option_payoff) * (option_payoff);
    *variance = sqrt(*variance / n);

    // Free the device memory
    gpuErrchk(cudaFree(d_total_squared_value));
    gpuErrchk(cudaFree(d_function));
    gpuErrchk(cudaFree(d_coefficients));
    gpuErrchk(cudaFree(d_assets_returns));
    gpuErrchk(cudaFree(d_assets_std_devs));
    gpuErrchk(cudaFree(d_assets_last_values));
    gpuErrchk(cudaFree(d_predicted_assets_prices));
    gpuErrchk(cudaFree(d_total_payoff));
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_zeta_matrix));


    std::cout << "--------->option payoff: " << option_payoff << std::endl;
    std::cout << "strike price: " << strike_price << std::endl;


    if (option_type == OptionType::European)
    {        
        for (size_t i = 0; i < assetPtrs.size(); ++i)
        {
            S += assetPtrs[i]->getLastRealValue();
            sigma += assetPtrs[i]->getReturnStdDev();
        }
        double d1 = (log(S / strike_price) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        BS_option_price = S * cuda_phi(d1) - strike_price * exp(-r * T) * cuda_phi(d2);

        std::cout << "The option price calculated via Black-Scholes model is " << BS_option_price << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cudaDeviceSynchronize();

    return std::make_pair(option_payoff, static_cast<double>(duration.count()));
}

