#include <cuda_runtime.h>
#include <curand_kernel.h>

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void generateGaussianNumbers( float *total_value, float *total_squared_value,
                                        const float *assets_returns, const float *assets_std_devs, long long int n,
                                        float *assets_closing_values, int strike_price, long long int seed, float *predicted_assets_prices )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO 4 is the maximum number of assets
    float result = 0.0;
    float rnd_daily_return = 0.0;
    float closing_value;

    if (tid < n)
    {
        curandState_t state;
        curand_init(seed, tid, 0, &state); // Moved outside the loop

        for( size_t asset_idx = 0; asset_idx < 4; asset_idx++)
        {
            closing_value = assets_closing_values[asset_idx];
            
            float return_value = curand_normal(&state);
            rnd_daily_return = assets_returns[asset_idx] + assets_std_devs[asset_idx] * return_value;
            closing_value = closing_value * (1 + rnd_daily_return);

            result += closing_value;
            atomicAdd(&predicted_assets_prices[asset_idx], closing_value);
        }

        if (result > strike_price)
            {
                result = result - strike_price;
            }else result = 0.0;

        atomicAdd(&total_value[tid % 100000], result);
        atomicAdd(&total_squared_value[tid % 100000], result * result);
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
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads_per_block = 256;
    dim3 number_of_blocks = (n + threads_per_block.x - 1) / threads_per_block.x;

    uint num_assets = assetPtrs.size();
    // Create and copy function and coefficients to device
    char *d_function;
    size_t function_size = function.size() + 1; // Include the null terminator
    gpuErrchk( cudaMalloc((void **)&d_function, function_size * sizeof(char)) );
    gpuErrchk( cudaMemcpy(d_function, function.c_str(), function_size * sizeof(char), cudaMemcpyHostToDevice) );

    double *d_coefficients;
    gpuErrchk( cudaMalloc((void **)&d_coefficients, coefficients.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(d_coefficients, coefficients.data(), coefficients.size() * sizeof(double), cudaMemcpyHostToDevice) );
    

    // Call the CUDA kernel to print the function and coefficients
    // printf("Calling CUDA kernel!\n");
     printFunction<<<1, 1>>>(n, d_function, d_coefficients, coefficients.size());
    // printf("CUDA kernel finished!\n");

    // Save the assets main data
    float *d_assets_returns;
    float *d_assets_std_devs;
    float *d_assets_last_values;
    gpuErrchk( cudaMalloc((void **)&d_assets_returns, num_assets * sizeof(float)) );
    gpuErrchk( cudaMalloc((void **)&d_assets_std_devs, num_assets * sizeof(float)) );
    gpuErrchk( cudaMalloc((void **)&d_assets_last_values, num_assets * sizeof(float)) );

    for (size_t i = 0; i < num_assets; i++)
    {
        float return_mean = static_cast<float>(assetPtrs[i]->getReturnMean());
        gpuErrchk( cudaMemcpy(&d_assets_returns[i], &return_mean, sizeof(float), cudaMemcpyHostToDevice) );

        float return_std_dev = static_cast<float>(assetPtrs[i]->getReturnStdDev());
        gpuErrchk( cudaMemcpy(&d_assets_std_devs[i], &return_std_dev, sizeof(float), cudaMemcpyHostToDevice) );

        float last_value = static_cast<float>(assetPtrs[i]->getLastRealValue());
        gpuErrchk( cudaMemcpy(&d_assets_last_values[i], &last_value, sizeof(float), cudaMemcpyHostToDevice) );
    }

    
    double total_value = 0.0;
    double total_squared_value = 0.0;

    float *d_total_value, *d_total_squared_value;
    gpuErrchk( cudaMalloc(&d_total_value, 100000 * sizeof(float)) );
    gpuErrchk( cudaMalloc(&d_total_squared_value, 100000 * sizeof(float)) );

    float predicted_assets_prices[num_assets];;
    float *d_predicted_assets_prices;
    gpuErrchk( cudaMalloc(&d_predicted_assets_prices, num_assets * sizeof(float)) );

    generateGaussianNumbers<<<number_of_blocks, threads_per_block>>>( d_total_value, d_total_squared_value, d_assets_returns, d_assets_std_devs, n, d_assets_last_values, strike_price, seed, d_predicted_assets_prices);
    cudaDeviceSynchronize();

    float host_total_value[100000];
    float host_total_squared_value[100000];
    gpuErrchk( cudaMemcpy(host_total_value, d_total_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(host_total_squared_value, d_total_squared_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost) );

    for (size_t i = 0; i <  100000; i++)
    {
        total_value += static_cast<double>( host_total_value[i] );
        total_squared_value += static_cast<double>( host_total_squared_value[i] );
    }


    float host_assets_prices[num_assets];
    gpuErrchk( cudaMemcpy(host_assets_prices, d_predicted_assets_prices, num_assets * sizeof(float), cudaMemcpyDeviceToHost) );

    for( size_t i = 0; i < num_assets; ++i )
    {
        predicted_assets_prices[i] = ( host_assets_prices[i]/n );
        std::cout << "The predicted future price (30 days) of one " << assetPtrs[i]->getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }


    double option_payoff = total_value / n;

    // calculate the variance
    *variance = total_squared_value/n - (total_value / n) * (total_value/ n);
    *variance = sqrt(*variance / static_cast<double>(n));


    // Free the device memory
    gpuErrchk( cudaFree(d_total_value) );
    gpuErrchk( cudaFree(d_total_squared_value) );
    gpuErrchk( cudaFree(d_function) );
    gpuErrchk( cudaFree(d_coefficients) );
    gpuErrchk( cudaFree(d_assets_returns) );
    gpuErrchk( cudaFree(d_assets_std_devs) );
    gpuErrchk( cudaFree(d_assets_last_values) );
    gpuErrchk( cudaFree(d_predicted_assets_prices) );
    // cudaFree(d_simulated_returns);

    printf("--------->option payoff: %f\n", option_payoff);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // return std::make_pair(69.0, 420.0);
    cudaDeviceSynchronize();

    return std::make_pair(option_payoff, static_cast<double>(duration.count()));
}