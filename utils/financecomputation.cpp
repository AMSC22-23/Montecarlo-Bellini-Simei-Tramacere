#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "../include/project/financecomputation.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/financemontecarlo.hpp"
#include "../include/project/optionparameters.hpp"
#include "../include/project/financeinputmanager.hpp"

void financeComputation()
{
    std::vector<Asset> assets;
    int csv_result = loadAssets("../data/", assets);
    if (csv_result == -1)
    {
        std::cout << "Error loading the assets from the CSV files" << std::endl;
        return;
    }
    std::vector<const Asset *> assetPtrs;
    for (const auto &asset : assets)
    {
        assetPtrs.push_back(&asset);
    }

    std::cout << "The assets have been loaded successfully. Calculating the price... " << std::endl;

    for (int n = 10; n < 1e9; n *= 10)
    {
        int    iterations        = n;
        int    strike_price      = calculateStrikePrice(assets);
        int    std_dev_from_mean = 24;
        double variance          = 0.0;

        auto function_pair = createPayoffFunction(strike_price, assets);
        auto function      = function_pair.first;
        auto coefficients  = function_pair.second;

        std::vector<double> integration_bounds(assets.size() * 2);
        if (getIntegrationBounds(integration_bounds, assets, std_dev_from_mean) == -1)
        {
            std::cout << "Error setting the integration bounds" << std::endl;
            return;
        }

        HyperRectangle hyperrectangle(assets.size(), integration_bounds);

        auto result = montecarloPricePrediction(iterations,
                                                function,
                                                hyperrectangle,
                                                assetPtrs,
                                                std_dev_from_mean,
                                                variance,
                                                coefficients,
                                                strike_price);

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

              // Write time and date to output.txt
            outputFile << "Generated on: " << formattedTimeStr.str() << "\n";
            outputFile << "=============================================================================================\n";
            outputFile << std::left << std::setw(20) << "Asset";
            outputFile << std::left << std::setw(20) << "Return Mean";
            outputFile << std::left << std::setw(20) << "Return Standard Deviation\n";
            for (const auto &asset : assets)
            {
                outputFile << std::left << std::setw(20) << asset.getName();
                outputFile << std::left << std::setw(20) << asset.getReturnMean();
                outputFile << std::left << std::setw(20) << asset.getReturnStdDev() << "\n";
            }
            outputFile << "=============================================================================================\n";
            outputFile << "The function is: " << function << "\n" << std::endl;
            outputFile << "The integration bounds are: " << "\n";
            for (size_t i = 0; i < integration_bounds.size(); i += 2)
            {
                outputFile << "[" << integration_bounds[i] << ", " << integration_bounds[i + 1] << "]\n";
            }
            outputFile << "=============================================================================================\n";
            outputFile << std::left << std::setw(25) << "Points";
            outputFile << std::left << std::setw(25) << "Error";
            outputFile << std::left << std::setw(25) << "E[option payoff]";
            outputFile << std::left << std::setw(25) << "Time[s]" << "\n";
        }

        outputFile << std::left << std::setw(25) << iterations;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << variance;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << result.first;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << result.second * 1e-6 << "\n";
        outputFile.close();
        std::cout << "The integral has been calculated successfully for " << iterations << " points" << std::endl;
    }

    std::cout << "\nThe results have been saved to output.txt" << std::endl;
}