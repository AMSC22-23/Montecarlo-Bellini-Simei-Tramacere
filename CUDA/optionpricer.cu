#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

extern "C++"
{
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/finance_computation.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/finance_montecarlo.hpp"
#include "../include/project/optionparameters.hpp"
#include "../include/project/finance_inputmanager.hpp"
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

double phi(double x)
{
    static const double RT2PI = sqrt(4.0 * acos(0.0));

    static const double SPLIT = 7.07106781186547;

    static const double N0 = 220.206867912376;
    static const double N1 = 221.213596169931;
    static const double N2 = 112.079291497871;
    static const double N3 = 33.912866078383;
    static const double N4 = 6.37396220353165;
    static const double N5 = 0.700383064443688;
    static const double N6 = 3.52624965998911e-02;
    static const double M0 = 440.413735824752;
    static const double M1 = 793.826512519948;
    static const double M2 = 637.333633378831;
    static const double M3 = 296.564248779674;
    static const double M4 = 86.7807322029461;
    static const double M5 = 16.064177579207;
    static const double M6 = 1.75566716318264;
    static const double M7 = 8.83883476483184e-02;

    const  double z = fabs(x);
    double c        = 0.0;

    if (z <= 37.0)
    {
        const double e = exp(-z * z / 2.0);
        if (z < SPLIT)
        {
            const double n = (((((N6 * z + N5) * z + N4) * z + N3) * z + N2) * z + N1) * z + N0;
            const double d = ((((((M7 * z + M6) * z + M5) * z + M4) * z + M3) * z + M2) * z + M1) * z + M0;
                  c        = e * n / d;
        }
        else
        {
            const double f = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))));
                  c        = e / (RT2PI * f);
        }
    }
    return x < = 0.0 ? c : 1 - c;
}

__global__ void generateGaussianNumbers(float *total_payoff, float *total_squared_value,
                                        const float *assets_returns, const float *assets_std_devs, long long int n,
                                        float *assets_closing_values, int strike_price, long long int seed, float *predicted_assets_prices)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
    {
          // TODO 4 is the maximum number of assets
        float payoff_function = 0.0;
        float dt              = 1.0 / 252 - 0;
        double prices[253];
        double r = 0.05;
        curandState_t state;
        curand_init(seed, tid, 0, &state);
        float rnv;
          // curand_init(seed, tid, 0, &state); // Moved outside the loop

        for (size_t asset_idx = 0; asset_idx < 4; asset_idx++)
        {

            prices[0] = assets_closing_values[asset_idx];

            for (uint step = 1; step < 253; ++step)
            {
                rnv = curand_normal(&state);
                  // printf("rnv %f\n", rnv);
                prices[step] = prices[step - 1] * exp((r - 0.5 * assets_std_devs[asset_idx] *
                                                                assets_std_devs[asset_idx]) *
                                                                dt +
                                                      assets_std_devs[asset_idx] * sqrt(dt) * rnv);
            }
            payoff_function += prices[252];  // option europea

            atomicAdd(&predicted_assets_prices[asset_idx], prices[252]);
            atomicAdd(&total_squared_value[asset_idx], prices[252]);
              // }
        }

        if (payoff_function > strike_price)
            payoff_function -= strike_price;
        else
            payoff_function = 0.0;

        atomicAdd(&total_payoff[0], payoff_function);
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

extern std::pair<double, double> kernel_wrapper(long long int n, const std::string &function, HyperRectangle &hyperrectangle,
                                                const std::vector<const Asset *> &assetPtrs, double std_dev_from_mean, double *variance,
                                                std::vector<double> coefficients, double strike_price, long long int seed)
{
    auto start             = std::chrono::high_resolution_clock::now();
    dim3 threads_per_block = 256;
    dim3 number_of_blocks  = (n + threads_per_block.x - 1) / threads_per_block.x;

    uint num_assets = assetPtrs.size();
      // Create and copy function and coefficients to device
    char *d_function;
    size_t function_size = function.size() + 1;  // Include the null terminator
    gpuErrchk(cudaMalloc((void **)&d_function, function_size * sizeof(char)));
    gpuErrchk(cudaMemcpy(d_function, function.c_str(), function_size * sizeof(char), cudaMemcpyHostToDevice));

    double *d_coefficients;
    gpuErrchk(cudaMalloc((void **)&d_coefficients, coefficients.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_coefficients, coefficients.data(), coefficients.size() * sizeof(double), cudaMemcpyHostToDevice));

      // Call the CUDA kernel to print the function and coefficients
    printFunction<<<1, 1>>>(n, d_function, d_coefficients, coefficients.size());

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
    gpuErrchk(cudaMalloc(&d_total_payoff, 1 * sizeof(float)));

    float predicted_assets_prices[num_assets];
    ;
    float *d_predicted_assets_prices;
    gpuErrchk(cudaMalloc(&d_predicted_assets_prices, num_assets * sizeof(float)));

    generateGaussianNumbers<<<number_of_blocks, threads_per_block>>>(d_total_payoff, d_total_squared_value, d_assets_returns, d_assets_std_devs, n, d_assets_last_values, strike_price, seed, d_predicted_assets_prices);
    cudaDeviceSynchronize();

    float host_total_squared_value[num_assets];
    gpuErrchk(cudaMemcpy(host_total_squared_value, d_total_squared_value, num_assets * sizeof(float), cudaMemcpyDeviceToHost));

    float host_assets_prices[num_assets];
    gpuErrchk(cudaMemcpy(host_assets_prices, d_predicted_assets_prices, num_assets * sizeof(float), cudaMemcpyDeviceToHost));

    float total_payoff[1];
    gpuErrchk(cudaMemcpy(&total_payoff, d_total_payoff, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    float total_squared_value = 0.0;

    for (size_t i = 0; i < num_assets; i++)
    {
        total_squared_value        += (host_total_squared_value[i]);
        predicted_assets_prices[i]  = (host_assets_prices[i] / n);
        std::cout << "The predicted future price (30 days) of one " << assetPtrs[i]->getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }

    float option_payoff = total_payoff[0] / n;

      // calculate the variance
    *variance = total_squared_value / n - (total_payoff[0] / n) * (total_payoff[0] / n);
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

    std::cout << "--------->option payoff: " << option_payoff << std::endl;
    std::cout << "strike price: " << strike_price << std::endl;

    double S               = 0.0;
    double r               = 0.05;
    double sigma           = 0.0;
    double T               = 1;
    double K               = strike_price;
    double BS_option_price = 0.0;

    for (size_t i = 0; i < assetPtrs.size(); ++i)
    {
        S     += assetPtrs[i]->getLastRealValue();
        sigma += assetPtrs[i]->getReturnStdDev();
    }
    double d1              = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2              = d1 - sigma * sqrt(T);
           BS_option_price = S * phi(d1) - K * exp(-r * T) * phi(d2);

    std::cout << "The option price calculated via Black-Scholes model is " << BS_option_price << std::endl;

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cudaDeviceSynchronize();

    return std::make_pair(option_payoff, static_cast<double>(duration.count()));
}