// CODICE CUDA FUNZIONANTE PRIMA DI PROVRE A IMPOSTARE prova_vector COME SHARED VARIABLE

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C++"
{
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/financecomputation.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/financemontecarlo.hpp"
#include "../include/project/optionparameters.hpp"
#include "../include/project/financeinputmanager.hpp"
}

// Parallelizzo sul numero di simulazioni, ma non sui 24 giorni considerati per ogni simulazione

__global__ void generateGaussianNumbers( float *total_value, float *total_squared_value,
                                        const float *assets_returns, const float *assets_std_devs, int num_days,
                                        long long int n, float *assets_closing_values, int strike_price, long long int seed
                                         /* ,float *min */)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float simulated_returns[4]; // TODO 4 is the maximum number of assets
    //size_t num_assets = sizeof(simulated_returns);
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
            
            // for (int i = 0; i < num_days; i++)
            // {
                float return_value = curand_normal(&state);
                rnd_daily_return = assets_returns[asset_idx] + assets_std_devs[asset_idx] * return_value;
                closing_value = closing_value * (1 + rnd_daily_return);
            // }

           

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
                result += closing_value;
                // atomicAdd(&min[tid+n*asset_idx], simulated_returns[asset_idx]);
                //printf("OK        Simulated return: %f, asset_idx: %d\n", simulated_returns[asset_idx], asset_idx);
             }
        }

        if (result > strike_price)
            {
                result = result - strike_price;
            }else result = 0.0;

        atomicAdd(&total_value[tid % 100000], result);
        atomicAdd(&total_squared_value[tid % 100000], result * result);
        // printf("Min: %f, %f, %f, %f\n", min[0], min[1], min[2], min[3]);
        // printf("Max: %f, %f, %f, %f\n", max[0], max[1], max[2], max[3]);
    }
}




__global__ void printFunction(long long int n, char *function, const double *coefficients, int number_of_coefficients)
{
    printf("Function: %s\n", function);
    printf("n: %d\n", n);
    for (size_t i = 0; i < number_of_coefficients; ++i)
    {
        printf("Coefficient[%d]: %f\n", i, coefficients[i]);
    }
    printf("Number of coefficients: %d\n", number_of_coefficients);
}



extern std::pair<double, double> kernel_wrapper(long long int n, const std::string &function, HyperRectangle &hyperrectangle,
                                                const std::vector<const Asset *> &assetPtrs /* = std::vector<const Asset*>() */,
                                                double std_dev_from_mean /* = 5.0 */, double *variance /* = nullptr */,
                                                std::vector<double> coefficients, double strike_price, long long int seed)
{
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads_per_block = 256;
    dim3 number_of_blocks = (n + threads_per_block.x - 1) / threads_per_block.x;


    // Create and copy function and coefficients to device
    char *d_function;
    size_t function_size = function.size() + 1; // Include the null terminator
    cudaMalloc((void **)&d_function, function_size * sizeof(char));
    cudaMemcpy(d_function, function.c_str(), function_size * sizeof(char), cudaMemcpyHostToDevice);

    double *d_coefficients;
    cudaMalloc((void **)&d_coefficients, coefficients.size() * sizeof(double));
    cudaMemcpy(d_coefficients, coefficients.data(), coefficients.size() * sizeof(double), cudaMemcpyHostToDevice);
    

    // Call the CUDA kernel to print the function and coefficients
    // printf("Calling CUDA kernel!\n");
    // printFunction<<<1, 1>>>(n, d_function, d_coefficients, coefficients.size());
    // printf("CUDA kernel finished!\n");

    // Save the assets main data
    float *d_assets_returns;
    float *d_assets_std_devs;
    float *d_assets_last_values;
    cudaMalloc((void **)&d_assets_returns, assetPtrs.size() * sizeof(float));
    cudaMalloc((void **)&d_assets_std_devs, assetPtrs.size() * sizeof(float));
    cudaMalloc((void **)&d_assets_last_values, assetPtrs.size() * sizeof(float));

    for (size_t i = 0; i < assetPtrs.size(); i++)
    {
        float return_mean = static_cast<float>(assetPtrs[i]->getReturnMean());
        cudaMemcpy(&d_assets_returns[i], &return_mean, sizeof(float), cudaMemcpyHostToDevice);

        float return_std_dev = static_cast<float>(assetPtrs[i]->getReturnStdDev());
        cudaMemcpy(&d_assets_std_devs[i], &return_std_dev, sizeof(float), cudaMemcpyHostToDevice);

        float last_value = static_cast<float>(assetPtrs[i]->getLastRealValue());
        cudaMemcpy(&d_assets_last_values[i], &last_value, sizeof(float), cudaMemcpyHostToDevice);
    }

    
    double total_value = 0.0;
    double total_squared_value = 0.0;

    // Generate random numbers: initialize the random number generator
    float *d_total_value, *d_total_squared_value;
    cudaMalloc(&d_total_value, 100000 * sizeof(float));
    cudaMalloc(&d_total_squared_value, 100000 * sizeof(float));
    int num_days = 24;

    // float values[assetPtrs.size() * n];
    // float *d_min;
    // cudaMalloc(&d_min, assetPtrs.size() * n * sizeof(float));
    // cudaMemset(d_min, 0, assetPtrs.size() * n * sizeof(float));
    // float min[assetPtrs.size()] = {1e9, 1e9, 1e9, 1e9};
    // float max[assetPtrs.size()] = {-1e9, -1e9, -1e9, -1e9};
    



    generateGaussianNumbers<<<number_of_blocks, threads_per_block>>>( d_total_value, d_total_squared_value, d_assets_returns, d_assets_std_devs, num_days, n, d_assets_last_values, strike_price, seed /*, d_min */);
    cudaDeviceSynchronize();

    // cudaMemcpy(values, d_min, assetPtrs.size() * n * sizeof(float), cudaMemcpyDeviceToHost);
    // for (size_t i = 0; i <n; ++i)
    // {
    //     for( size_t j = 0; j < assetPtrs.size()*n; j+=n)
    //     {
    //         if( values[i+j]<min[j/n])
    //             min[j/n] = values[i+j];
    //         if( values[i+j]>max[j/n])
    //             max[j/n] = values[i+j];
    //     }
    // }

    // double domain = 1.0;
    // for( size_t i = 0; i < assetPtrs.size(); i++)
    // {
    //     printf("Min[%d]: %f\n", i, min[i]);
    //     printf("Max[%d]: %f\n", i, max[i]);
    //     domain = domain * (max[i] - min[i]);
    // }

    double domain = 1.0;
    double integration_bounds[assetPtrs.size() * 2 - 1];
    int j = 0;

        for (size_t i = 0; i < assetPtrs.size() * 2 - 1; i += 2)
        {
            integration_bounds[i]     = assetPtrs[j]->getReturnMean() - 24 * assetPtrs[j]->getReturnStdDev() + 1.0;
            integration_bounds[i + 1] = assetPtrs[j]->getReturnMean() + 24 * assetPtrs[j]->getReturnStdDev() + 1.0;
            j++;
            domain *= (integration_bounds[i + 1] - integration_bounds[i]);
        }
    // printf("Domain: %f\n", domain);



    float hostTotalValue[100000];
    float hostTotalSquaredValue[100000];
    cudaMemcpy(hostTotalValue, d_total_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTotalSquaredValue, d_total_squared_value, 100000 * sizeof(float), cudaMemcpyDeviceToHost);
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


    hyperrectangle.calculateVolume();
    double domain2 = hyperrectangle.getVolume();
    //printf("Domain: %f\n", domain);
    // printf("Domain old: %f\n", domain2);
    double integral = total_value / n * domain;

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
    // cudaFree(d_simulated_returns);

    printf("--------->Integral: %f\n", integral);
    // printf("n: %d\n", n);
    // printf("Variance: %f\n", *variance);
    //   printf( "total_value: %f\n", total_value);
    // printf("total_squared_value: %f\n", total_squared_value);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // return std::make_pair(69.0, 420.0);
    cudaDeviceSynchronize();

    return std::make_pair(integral, static_cast<double>(duration.count()));
}