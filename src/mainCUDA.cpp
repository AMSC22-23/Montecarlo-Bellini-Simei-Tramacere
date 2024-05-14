#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "../include/muparser-2.3.4/include/muParser.h"
#include "../include/muparser-2.3.4/include/muParserIncluder.h"

#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/montecarlointegrator.hpp"
#include "../include/project/csvhandler.hpp"

extern std::pair<double, double> kernel_wrapper(long long int N, const std::string &function, HyperRectangle &hyperrectangle,
                                                const std::vector<const Asset *> &assetPtrs /* = std::vector<const Asset*>() */,
                                                double std_dev_from_mean /* = 5.0 */, double *variance /* = nullptr */,
                                                std::vector<double> coefficients, int strike_price);
// extern std::pair<double, double> kernel_wrapper();

int main(int argc, char **argv)
{
  using namespace std::chrono;

  // long long int N = 4194304;
  long long int N = 1e8;
  int strike_price = 0;
  int std_dev_from_mean = 24;
  double variance = 0.0;

  // Load the assets from the csv file
  std::vector<Asset> assets;
  int csv_result = load_assets_from_csv("../data/", assets);
  if (csv_result == -1)
  {
    std::cout << "Error loading the assets from the CSV files" << std::endl;
    return -1;
  }
  std::vector<const Asset *> assetPtrs;
  for (const auto &asset : assets)
  {
    assetPtrs.push_back(&asset);
  }

  printf("Pricing the option ...\n");
  // Get starting timepoint
  auto start = high_resolution_clock::now();
  strike_price = calculate_strike_price(assets);

  // std::string function = create_function(strike_price, assets);
  std::pair<std::string, std::vector<double>> function_pair = create_function(strike_price, assets);
  auto function = function_pair.first;
  auto coefficients = function_pair.second;

  std::vector<double> integration_bounds(assets.size() * 2);
  if (set_integration_bounds(integration_bounds, assets, std_dev_from_mean) == -1)
  {
    std::cout << "Error setting the integration bounds" << std::endl;
    return -1;
  }

  HyperRectangle hyperrectangle(assets.size(), integration_bounds);

  // std::pair<double, double> result = kernel_wrapper(N, function, hyperrectangle, true, assetPtrs, std_dev_from_mean, &variance);
  std::pair<double, double> result = kernel_wrapper(N, function, hyperrectangle, assetPtrs, std_dev_from_mean, &variance,
                                                    coefficients, strike_price);

  // Open the output file stream
  std::ofstream outputFile("output.txt", std::ios::app);

  // Check if the file is empty
  outputFile.seekp(0, std::ios::end);
  bool isEmpty = (outputFile.tellp() == 0);
  outputFile.seekp(0, std::ios::beg);

  // Write the title of the columns if the file is empty
  if (isEmpty)
  {
    // Print asset information to the file in a table format
    outputFile << std::left << std::setw(20) << "Asset";
    outputFile << std::left << std::setw(20) << "Return Mean";
    outputFile << std::left << std::setw(20) << "Return Standard Deviation\n";

    for (const auto &asset : assets)
    {
      outputFile << std::left << std::setw(20) << asset.get_name();
      outputFile << std::left << std::setw(20) << asset.get_return_mean();
      outputFile << std::left << std::setw(20) << asset.get_return_std_dev() << "\n";
    }
    outputFile << "=============================================================================================\n";

    // Write additional information to the file
    outputFile << "The function is: " << function << "\n";
    outputFile << "The integration bounds are: "
               << "\n";
    for (size_t i = 0; i < integration_bounds.size(); i += 2)
    {
      outputFile << "[" << integration_bounds[i] << ", " << integration_bounds[i + 1] << "]\n";
    }
    outputFile << "=============================================================================================\n";

    outputFile << std::left << std::setw(15) << "Points";
    outputFile << std::left << std::setw(15) << "Variance";
    outputFile << std::left << std::setw(15) << "Final Price";
    outputFile << std::left << std::setw(15) << "Time"
               << "\n";
  }

  // Write important data to file with increased column width
  outputFile << std::left << std::setw(15) << N;
  outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << variance;
  outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.first;
  outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.second * 1e-6 << "\n";

  // Close the output file stream
  outputFile.close();

  auto stop = high_resolution_clock::now();
  // Get duration. Substart timepoints to
  // get duration. To cast it to proper unit
  // use duration cast method
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time taken by function: "
            << duration.count() << " milliseconds" << std::endl;

  printf("Done\n");
  printf("\n");

  return 0;
}