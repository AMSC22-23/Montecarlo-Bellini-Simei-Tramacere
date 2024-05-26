#include "../include/project/finance_montecarlo.hpp"

std::pair<double, double> montecarloPricePrediction(int points, const std::string &function, HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double std_dev_from_mean, double &variance,
                                                    std::vector<double> coefficients, double strike_price, std::vector<double> &predicted_assets_prices)
{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    auto   start               = std::chrono::high_resolution_clock::now();

// #pragma omp parallel {
    for (size_t i = 0; i < assetPtrs.size(); ++i)
    {
        predicted_assets_prices[i] = 0;
    }
// }

#pragma omp parallel
    {
        double total_value_thread         = 0.0;
        double total_squared_value_thread = 0.0;
        std::vector<double> random_point_vector(assetPtrs.size());
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (size_t i = 0; i < points; ++i)
        {

            hyperrectangle.financeGenerateRandomPoint(random_point_vector, assetPtrs, std_dev_from_mean, predicted_assets_prices);

            if (!random_point_vector.empty())
            {

                result = 0.0;
                for (size_t i = 0; i < random_point_vector.size(); ++i)
                {
                    result += random_point_vector[i] * coefficients[i];
                    predicted_assets_prices[i] += result;
                }

                result = std::max(0.0, (result - strike_price));

                total_value_thread         += result;
                total_squared_value_thread += result * result;
            }
            else
            {
                std::cout << "Error generating random point" << std::endl;
                i--;
            }
        }

#pragma omp critical
        {
            total_value         += total_value_thread;
            total_squared_value += total_squared_value_thread;
        }
    }

      // calculate the integral
    hyperrectangle.calculateVolume();
    double domain   = hyperrectangle.getVolume();
    double integral = total_value / static_cast<double>(points) * domain;

      // calculate the variance
    variance = total_squared_value / static_cast<double>(points) - (total_value / static_cast<double>(points)) * (total_value / static_cast<double>(points));
    variance = sqrt(variance / static_cast<double>(points)); //! transfer in the main

      // stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, static_cast<double>(duration.count()));
}