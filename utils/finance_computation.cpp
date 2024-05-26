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

  size_t num_iterations = 10;
  int iterations = 1e6;
  double strike_price = calculateStrikePrice(assets);
  int std_dev_from_mean = 24;
  double variance = 0.0;
  std::pair<double, double> result;
  std::pair<double, double> result_temp;
  result.first = 0.0;
  result.second = 0.0;

  auto function_pair = createPayoffFunction(strike_price, assets);
  auto function = function_pair.first;
  auto coefficients = function_pair.second;

  std::vector<double> integration_bounds(assets.size() * 2);
  std::cout << "Calculating the price of the option..." << std::endl;
  for (size_t j = 0; j < num_iterations; j++)
  {

    if (getIntegrationBounds(integration_bounds, assets, std_dev_from_mean) == -1)
    {
      std::cout << "Error setting the integration bounds" << std::endl;
      return;
    }

    HyperRectangle hyperrectangle(assets.size(), integration_bounds);

    result_temp = montecarloPricePrediction(iterations,
                                            function,
                                            hyperrectangle,
                                            assetPtrs,
                                            std_dev_from_mean,
                                            variance,
                                            coefficients,
                                            strike_price);
    result.first += result_temp.first;
    result.second += result_temp.second;
    double progress = static_cast<double>( j + 1 )/ static_cast<double>(num_iterations) * 100;
    std::cout << "Process at " << progress << "% ...\n";
  }
  result.first /= num_iterations;
  variance /= num_iterations;

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
    outputFile << "The function is: " << function << "\n"
               << std::endl;
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
  std::cout << "The integral has been calculated successfully for " << iterations << " points" << std::endl << "and " << num_iterations << " times, averaging its value." << std::endl;

  std::cout << "\nThe results have been saved to output.txt" << std::endl;
}
