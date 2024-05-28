#include "../include/project/finance_montecarlo.hpp"


// Function to calculate the option price prediction using the Monte Carlo method
// The function is the core of the finance oriented project, which is used to predict 
// the option price prediction using the Monte Carlo method.
std::pair<double, double> montecarloPricePrediction(size_t points,
                                                    const std::string &function,
                                                    HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    const double std_dev_from_mean,
                                                    double &variance,
                                                    std::vector<double> coefficients,
                                                    const double strike_price,
                                                    std::vector<double> &predicted_assets_prices)
{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    auto   start               = std::chrono::high_resolution_clock::now();

// Parallelize the Monte Carlo method for the finance project 
// by using OpenMP to speed up the computation
// The points are generated by parallelizing the generation for each asset
// The random point vector is used to store the value of each asset 
// of the random generated point
#pragma omp parallel
    {
        double total_value_thread         = 0.0;
        double total_squared_value_thread = 0.0;
        std::vector<double> random_point_vector(assetPtrs.size(), 0.0);
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < points; ++i)
        {

            hyperrectangle.financeGenerateRandomPoint(random_point_vector, assetPtrs, std_dev_from_mean);
            
            // Check if the random point vector is not null
            if (random_point_vector[0] != 0.0)
            {

                result = 0.0;

                // Evaluate the payoff function with the random point
                for (size_t i = 0; i < random_point_vector.size(); ++i)
                {
                    result += random_point_vector[i] * coefficients[i];
                    predicted_assets_prices[i] += random_point_vector[i] * coefficients[i];
                }

                result = std::max(0.0, (result - strike_price));

                total_value_thread         += result;
                total_squared_value_thread += result * result;
            }
            else
            {
                std::cout << "Error generating random point." << std::endl;
                i--;
            }
        }

#pragma omp critical
        {
            total_value         += total_value_thread;
            total_squared_value += total_squared_value_thread;
        }
    }

    // Calculate the integral
    double option_payoff = total_value / static_cast<double>(points);

    // Calculate the variance
    variance = total_squared_value / static_cast<double>(points) - (total_value / static_cast<double>(points)) * (total_value / static_cast<double>(points));

    // Stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(option_payoff, static_cast<double>(duration.count()));
}