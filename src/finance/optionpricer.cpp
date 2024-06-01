#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "../../include/project/finance_computation.hpp"
#include "../../include/project/asset.hpp"
#include "../../include/project/finance_montecarlo.hpp"
#include "../../include/project/optionparameters.hpp"
#include "../../include/project/finance_inputmanager.hpp"

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

    const double z = fabs(x);
    double c = 0.0;

    if (z <= 37.0)
    {
        const double e = exp(-z * z / 2.0);
        if (z < SPLIT)
        {
            const double n = (((((N6 * z + N5) * z + N4) * z + N3) * z + N2) * z + N1) * z + N0;
            const double d = ((((((M7 * z + M6) * z + M5) * z + M4) * z + M3) * z + M2) * z + M1) * z + M0;
            c = e * n / d;
        }
        else
        {
            const double f = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))));
            c = e / (RT2PI * f);
        }
    }
    return x <= 0.0 ? c : 1 - c;
}

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

    size_t num_iterations = 10;
    size_t num_simulations = 1e5;
    double strike_price = calculateStrikePrice(assets);
    const uint std_dev_from_mean = 24;
    double variance = 0.0;
    double variance_temp = 0.0;
    double standard_error = 0.0;
    std::pair<double, double> result;
    std::pair<double, double> result_temp;
    result.first = 0.0;
    result.second = 0.0;

    // Create the payoff function.
    // The payoff function describes the financial beahviour of the option,
    // and it is required to calculate the price of the option.
    auto function_pair = createPayoffFunction(strike_price, assets);
    auto function = function_pair.first;
    auto coefficients = function_pair.second;
    std::vector<double> predicted_assets_prices;
    predicted_assets_prices.resize(assets.size());
    double BS_option_price = 0.0;
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

    for (size_t i = 0; i < assets.size(); i++)
    {
        std::cout << "mean and std dev of " << assets[i].getName() << " is " << assets[i].getReturnMean() << " and " << assets[i].getReturnStdDev() << std::endl;
    }
    std::cout << "Calculating the price of the option..." << std::endl;

    // Apply the Monte Carlo method to calculate the price of the option
    for (size_t j = 0; j < num_iterations; ++j)
    {
        result_temp = montecarloPricePrediction(num_simulations,
                                                function,
                                                hyperrectangle,
                                                assetPtrs,
                                                std_dev_from_mean,
                                                variance_temp,
                                                coefficients,
                                                strike_price,
                                                predicted_assets_prices);

        result.first += result_temp.first;
        result.second += result_temp.second;
        variance += variance_temp;
        standard_error += std::sqrt(variance_temp / static_cast<double>(num_simulations));

        double progress = static_cast<double>(j + 1) / static_cast<double>(num_iterations) * 100;
        std::cout << "Process at " << progress << "% ...\n";
    }

    result.first /= num_iterations;
    variance /= num_iterations;
    standard_error /= num_iterations;

    for (size_t i = 0; i < assets.size(); ++i)
    {
        predicted_assets_prices[i] /= (num_iterations * num_simulations/2);
    }

    double S = 0.0;
    double r = 0.05;
    double sigma = 0.0;
    double T = 1;
    double K = strike_price;

    for (size_t i = 0; i < assets.size(); ++i)
    {
        S += assetPtrs[i]->getLastRealValue();
        sigma += assetPtrs[i]->getReturnStdDev();
    }
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    BS_option_price = S * phi(d1) - K * exp(-r * T) * phi(d2);




    std::cout << "The option price calculated via Black-Scholes model is " << BS_option_price << std::endl;

    std::cout << "The option price calculated via Monte Carlo method is " << result.first << std::endl;

    std::ofstream outputFile("output.txt", std::ios::app);
    outputFile.seekp(0, std::ios::end);
    bool isEmpty = (outputFile.tellp() == 0);
    outputFile.seekp(0, std::ios::beg);

    if (isEmpty)
    {
        // Get current time and date
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        // Convert time to string
        std::string timeStr = std::ctime(&time);
        timeStr.pop_back(); // Remove the newline character at the end

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
        outputFile << "==================================================================================================================================\n";
        outputFile << std::left << std::setw(22) << "Points";
        outputFile << std::left << std::setw(22) << "Error";
        outputFile << std::left << std::setw(22) << "Option payoff";
        outputFile << std::left << std::setw(22) << "Time[s]" << "\n";
    }

    // Write the results to output.txt
    outputFile << std::left << std::setw(22) << num_simulations;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << standard_error;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.first;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.second * 1e-6 << "\n";
    outputFile.close();
    std::cout << "\nThe integral has been calculated successfully " << num_iterations << " times for " << num_simulations << " points." << std::endl;
    std::cout << "The resulting expected option payoff is the average of the " << num_iterations << " num_simulations.\n";
    std::cout << "\nThe results have been saved to output.txt\n"
              << std::endl;

    for (size_t i = 0; i < assets.size(); ++i)
    {
        std::cout << "The predicted future prices (30 days) of one " << assets[i].getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }
}