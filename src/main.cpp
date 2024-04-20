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

int main()
{
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

    std::cout << "The assets have been loaded successfully. Calculating the price... " << std::endl;

    // Loop through the number of iterations
    for (int n = 10; n < 1e9; n *= 10)
    {
        // Predict the future month for each asset
        int iterations = n;

        int strike_price = calculate_strike_price(assets);
        // int strike_price = 0;

        int std_dev_from_mean = 24;

        double variance = 0.0;

        std::pair<std::string,std::vector<double>> function_pair = create_function(strike_price, assets);
        auto function = function_pair.first;
        auto coefficients = function_pair.second;

        std::vector<double> integration_bounds(assets.size() * 2);
        if (set_integration_bounds(integration_bounds, assets, std_dev_from_mean) == -1)
        {
            std::cout << "Error setting the integration bounds" << std::endl;
            return -1;
        }

        HyperRectangle hyperrectangle(assets.size(), integration_bounds);

        std::pair<double, double> result = Montecarlo_integration(iterations, function, hyperrectangle, 
                                                                  true, assetPtrs, std_dev_from_mean, &variance, 
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
            
            outputFile << std::left << std::setw(25) << "Points";
            outputFile << std::left << std::setw(25) << "Variance";
            outputFile << std::left << std::setw(25) << "E[option payoff]";
            outputFile << std::left << std::setw(25) << "Time"
                       << "\n";
        }

          // Write important data to file with increased column width
        outputFile << std::left << std::setw(25) << iterations;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << variance;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << result.first;
        outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(25) << result.second * 1e-6 << "\n";

          // Close the output file stream
        outputFile.close();
        std::cout << "The integral has been calculated successfully for " << iterations << " points" << std::endl;
    }

      /*std::cout << "===============================================================================================================" << std::endl;
std::cout << std::endl;

int n, dim;
double rad, edge;
std::string function;
std::string domain_type;
std::vector<double> hyper_rectangle_bounds;
std::pair<double, double> result = {0.0, 0.0};

      // get the input from the user
input_manager(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

if (domain_type == "hs")
{
  HyperSphere hypersphere(dim, rad);
  if (function == "1")
  {
      hypersphere.calculate_volume();
      result.first = hypersphere.get_volume();
  }
  else
  {
      result = Montecarlo_integration(n, function, hypersphere);
  }
}
else if (domain_type == "hc")
{
  HyperCube hypercube(dim, edge);
  if (function == "1")
  {
      hypercube.calculate_volume();
      result.first = hypercube.get_volume();
  }
  else
  {
      result = Montecarlo_integration(n, function, hypercube);
  }
}
else if (domain_type == "hr")
{
  HyperRectangle hyperrectangle(dim, hyper_rectangle_bounds);
  if (function == "1")
  {
      hyperrectangle.calculate_volume();
      result.first = hyperrectangle.get_volume();
  }
  else
  {
      result = Montecarlo_integration(n, function, hyperrectangle);
  }
}

std::cout << std::endl
        << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
if (result.second != 0.0)
  std::cout << "The time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
*/
    return 0;
}
