#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "../include/project/finance_computation.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/finance_montecarlo.hpp"
#include "../include/project/optionparameters.hpp"
#include "../include/project/finance_inputmanager.hpp"


// Function that embeds multiple methods that are used to compute 
// the option price using the Monte Carlo method
void financeComputation()
{
    std::vector<Asset> assets;

    // Load the assets from the CSV files
    std::cout << "Loading assets from csv..." << std::endl;

    int csv_result = loadAssets("../data/", assets);
    if (csv_result == -1)
    {
        std::cout << "Error loading the assets from the CSV files" << std::endl;
        return;
    }

    std::vector<const Asset *> assetPtrs;
    assetPtrs.reserve(assets.size());
    for (const auto &asset : assets)
    {
        assetPtrs.emplace_back(&asset);
    }

    std::cout << "The assets have been loaded successfully." << std::endl;

    size_t num_iterations         = 10;
    size_t iterations             = 1e5;
    double strike_price           = calculateStrikePrice(assets);
    const  uint std_dev_from_mean = 24;
    double variance               = 0.0;
    double variance_temp          = 0.0;
    double standard_error         = 0.0;
    std::pair<double, double> result;
    std::pair<double, double> result_temp;
    result.first  = 0.0;
    result.second = 0.0;

    // Create the payoff function.
    // The payoff function describes the financial beahviour of the option,
    // and it is required to calculate the price of the option.
    auto function_pair = createPayoffFunction(strike_price, assets);
    auto function      = function_pair.first;
    auto coefficients  = function_pair.second;
    std::vector<double> predicted_assets_prices;
    predicted_assets_prices.resize(assets.size());

    std::vector<double> integration_bounds;
    integration_bounds.resize(assets.size() * 2);

    // Set the integration bounds based on the assets, on which the domain of the hyperrectangle is based.
    // The integration bounds are required in order to apply the Monte Carlo method for the option pricing,
    // and they are calculated based on the standard deviation from the mean of the assets.
    if (getIntegrationBounds(integration_bounds, assets, std_dev_from_mean) == -1)
    {
        std::cout << "Error setting the integration bounds" << std::endl;
        return;
    }

    HyperRectangle hyperrectangle(assets.size(), integration_bounds);

    std::cout << "Calculating the price of the option..." << std::endl;

    // Apply the Monte Carlo method to calculate the price of the option
    for (size_t j = 0; j < num_iterations; ++j)
    {
        result_temp = montecarloPricePrediction(iterations,
                                                function,
                                                hyperrectangle,
                                                assetPtrs,
                                                std_dev_from_mean,
                                                variance_temp,
                                                coefficients,
                                                strike_price,
                                                predicted_assets_prices);

        result.first   += result_temp.first;
        result.second  += result_temp.second;
        variance       += variance_temp;
        standard_error += std::sqrt(variance_temp / static_cast<double>(iterations)) * hyperrectangle.getVolume();

        double progress = static_cast<double>(j + 1) / static_cast<double>(num_iterations) * 100;
        std::cout << "Process at " << progress << "% ...\n";
    }

    result.first   /= num_iterations;
    variance       /= num_iterations;
    standard_error /= num_iterations;

    for (size_t i = 0; i < assets.size(); ++i)
    {
        predicted_assets_prices[i] /= (num_iterations * iterations);
    }

    std::ofstream outputFile("output.txt", std::ios::app);
    outputFile.seekp(0, std::ios::end);
    bool isEmpty = (outputFile.tellp() == 0);
    outputFile.seekp(0, std::ios::beg);

    if (isEmpty)
    {
        // Get current time and date
        auto        now  = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        // Convert time to string
        std::string timeStr = std::ctime(&time);
        timeStr.pop_back();  // Remove the newline character at the end

        // Parse the date and time string
        std::istringstream ss(timeStr);
        std::string dayOfWeek, month, day, timeOfDay, year;
        ss >> dayOfWeek >> month >> day >> timeOfDay >> year;

        // Reformat the string to have the year before the time
        std::ostringstream formattedTimeStr;
        formattedTimeStr << dayOfWeek << " " << month << " " << day << " " << year << " " << timeOfDay;

        // Write the results to output.txt
        outputFile << "Generated on: " << formattedTimeStr.str() << "\n";
        outputFile << "==================================================================================================================================\n";
        outputFile << std::left << std::setw(22) << "Asset";
        outputFile << std::left << std::setw(22) << "Return Mean";
        outputFile << std::left << std::setw(22) << "Return Standard Deviation\n";
        for (const auto &asset : assets)
        {
            outputFile << std::left << std::setw(22) << asset.getName();
            outputFile << std::left << std::setw(22) << asset.getReturnMean();
            outputFile << std::left << std::setw(22) << asset.getReturnStdDev() << "\n";
        }
        outputFile << "==================================================================================================================================\n";
        outputFile << "The function is: " << function << "\n"
                   << std::endl;
        outputFile << "The integration bounds are: " << "\n";
        for (size_t i = 0; i < integration_bounds.size(); i += 2)
        {
            outputFile << "[" << integration_bounds[i] << ", " << integration_bounds[i + 1] << "]\n";
        }
        outputFile << "==================================================================================================================================\n";
        outputFile << std::left << std::setw(22) << "Points";
        outputFile << std::left << std::setw(22) << "Error";
        outputFile << std::left << std::setw(22) << "Variance";
        outputFile << std::left << std::setw(22) << "E[option payoff]";
        outputFile << std::left << std::setw(22) << "95% conf. interval";
        outputFile << std::left << std::setw(22) << "Time[s]" << "\n";
    }

    // Calculate the confidence interval, 
    // with a critical value of 1.96 for a 95% confidence interval
    double critical_value = 1.96;
    double margin_of_error = critical_value * standard_error;
    double lower_bound     = result.first - margin_of_error;
    double upper_bound     = result.first + margin_of_error;

    // Write the results to output.txt
    outputFile << std::left << std::setw(22) << iterations;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << standard_error;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << variance;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.first;
    outputFile << std::fixed << std::setprecision(6) << std::left << "[" << lower_bound << ", " << upper_bound << std::setw(3) << "]";
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.second * 1e-6 << "\n";
    outputFile.close();
    std::cout << "\nThe integral has been calculated successfully " << num_iterations << " times for " << iterations << " points." << std::endl;
    std::cout << "The resulting expected option payoff is the average of the " << num_iterations << " iterations.\n";
    std::cout << "\nThe results have been saved to output.txt\n"
              << std::endl;

    for (size_t i = 0; i < assets.size(); ++i)
    {
        std::cout << "The predicted future prices (30 days) of one " << assets[i].getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }
}