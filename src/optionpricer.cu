// CODICE CUDA FUNZIONANTE PRIMA DI PROVRE A IMPOSTARE prova_vector COME SHARED VARIABLE

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

extern "C++"
{
#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/montecarlointegrator.hpp"
#include "../include/project/csvhandler.hpp"
}

// Parallelizzo sul numero di simulazioni, ma non sui 24 giorni considerati per ogni simulazione

__global__ void generateGaussianNumbers( float *total_value, float *total_squared_value,
                                        const float *assets_returns, const float *assets_std_devs, int num_days,
                                        long long int n, float *assets_closing_values, int strike_price)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float simulated_returns[4]; // TODO 4 is the maximum number of assets
    double result = 0.0;
    double rnd_daily_return = 0.0;
    float closing_value;

    if (tid < n)
    {
        for( int asset_idx = 0; asset_idx < 4; asset_idx++)
        {
            closing_value = assets_closing_values[asset_idx];
            curandState_t state;
            curand_init(tid%100, 0, 0, &state); // Moved outside the loop
            for (int i = 0; i < num_days; i++)
            {
                double return_value = curand_normal(&state);
                rnd_daily_return = assets_returns[asset_idx] + assets_std_devs[asset_idx] * return_value;
                closing_value = closing_value * (1 + rnd_daily_return);
            }
            simulated_returns[asset_idx] = closing_value/assets_closing_values[asset_idx];

            if (simulated_returns[asset_idx] < assets_returns[asset_idx] - 24 * assets_std_devs[asset_idx] + 1.0){
                //printf("Simulated return out of bounds: %f < %f\n", simulated_returns[asset_idx], assets_returns[asset_idx] - 24 * assets_std_devs[asset_idx] + 1.0);
                asset_idx--;                
                continue;

            } else if(simulated_returns[asset_idx] > assets_returns[asset_idx] + 24 * assets_std_devs[asset_idx] + 1.0 ){
                //printf("Simulated return out of bounds: %f > %f\n", simulated_returns[asset_idx], assets_returns[asset_idx] + 24 * assets_std_devs[asset_idx] + 1.0);
                asset_idx--;                
                continue;
            }
            else {
                result += simulated_returns[asset_idx]*assets_closing_values[asset_idx];
                //printf("OK        Simulated return: %f, asset_idx: %d\n", simulated_returns[asset_idx], asset_idx);
             }
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
    printf("n: %d\n", n);
    for (int i = 0; i < number_of_coefficients; ++i)
    {
        printf("Coefficient[%d]: %f\n", i, coefficients[i]);
    }
    printf("Number of coefficients: %d\n", number_of_coefficients);
}



extern std::pair<double, double> kernel_wrapper(long long int n, const std::string &function, HyperRectangle &hyperrectangle,
                                                const std::vector<const Asset *> &assetPtrs /* = std::vector<const Asset*>() */,
                                                double std_dev_from_mean /* = 5.0 */, double *variance /* = nullptr */,
                                                std::vector<double> coefficients, int strike_price)
{
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads_per_block = 256;
    dim3 number_of_blocks = (n + threads_per_block.x - 1) / threads_per_block.x;


    // Create and copy function and coefficients to device
    char *d_function;
    size_t function_size = function.size() + 1; // Include the null terminator
    cudaMalloc((void **)&d_function, function_size * sizeof(char));
    cudaDeviceSynchronize();
    cudaMemcpy(d_function, function.c_str(), function_size * sizeof(char), cudaMemcpyHostToDevice);

    double *d_coefficients;
    cudaMalloc((void **)&d_coefficients, coefficients.size() * sizeof(double));
    cudaDeviceSynchronize();
    cudaMemcpy(d_coefficients, coefficients.data(), coefficients.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Call the CUDA kernel to print the function and coefficients
    printf("Calling CUDA kernel!\n");
    printFunction<<<1, 1>>>(n, d_function, d_coefficients, coefficients.size());
    cudaDeviceSynchronize();
    printf("CUDA kernel finished!\n");

    // Save the assets main data
    float *d_assets_returns;
    float *d_assets_std_devs;
    float *d_assets_last_values;
    cudaMalloc((void **)&d_assets_returns, assetPtrs.size() * sizeof(float));
    cudaMalloc((void **)&d_assets_std_devs, assetPtrs.size() * sizeof(float));
    cudaMalloc((void **)&d_assets_last_values, assetPtrs.size() * sizeof(float));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < assetPtrs.size(); i++)
    {
        float return_mean = static_cast<float>(assetPtrs[i]->get_return_mean());
        cudaMemcpy(&d_assets_returns[i], &return_mean, sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        float return_std_dev = static_cast<float>(assetPtrs[i]->get_return_std_dev());
        cudaMemcpy(&d_assets_std_devs[i], &return_std_dev, sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        float last_value = static_cast<float>(assetPtrs[i]->get_last_real_value());
        cudaMemcpy(&d_assets_last_values[i], &last_value, sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    
    int num_assets = assetPtrs.size();
    double total_value = 0.0;
    double total_squared_value = 0.0;

    // Generate random numbers: initialize the random number generator
    float *d_total_value, *d_total_squared_value;
    cudaMalloc(&d_total_value, 100000 * sizeof(float));
    cudaMalloc(&d_total_squared_value, 100000 * sizeof(float));
    cudaDeviceSynchronize();
    int num_days = 24;

    // Call the CUDA kernel to simulate the assets returns over the next 24 working days, for each asset
    // for (size_t i = 0; i < num_assets; i++)
    // {
        generateGaussianNumbers<<<number_of_blocks, threads_per_block>>>( d_total_value, d_total_squared_value, d_assets_returns, d_assets_std_devs, num_days, n, d_assets_last_values, strike_price);
        cudaDeviceSynchronize();        
    // }
    

    float hostTotalValue[100000];
    float hostTotalSquaredValue[100000];
    cudaMemcpy(hostTotalValue, d_total_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTotalSquaredValue, d_total_squared_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Print the values
    for (size_t i = 0; i <  100000; i++)
    {
        total_value += static_cast<double>( hostTotalValue[i] );
        total_squared_value += static_cast<double>( hostTotalSquaredValue[i] );
        
        if( i<= n ){
            // printf( "total_value [%d] = %f \n" , i, total_value );
            // printf( "total_squared_value [%d] = %f \n" , i, total_squared_value );
        }
    }


    hyperrectangle.calculate_volume();
    double domain = hyperrectangle.get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

    // calculate the variance
    *variance = total_squared_value/n - (total_value / n) * (total_value/ n);
    *variance = sqrt(*variance / static_cast<double>(n));


    // Free the device memory
    cudaFree(d_total_value);
    cudaFree(d_total_squared_value);
    cudaFree(d_function);
    cudaFree(d_coefficients);
    cudaFree(d_assets_returns);
    cudaFree(d_assets_std_devs);
    cudaFree(d_assets_last_values);

    printf("Integral: %f\n", integral);
    printf("Variance: %f\n", *variance);
    printf( "total_value: %f\n", total_value);
    printf("total_squared_value: %f\n", total_squared_value);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // return std::make_pair(69.0, 420.0);
    //  return std::make_pair(69.0, 420.0);

    return std::make_pair(integral, static_cast<double>(duration.count()));
}