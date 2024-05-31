#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "../../include/project/finance_inputmanager.hpp"


// Function to get the integration bounds for the Monte Carlo method
// of the finance oriented project
int getIntegrationBounds(std::vector<double> &integration_bounds,
                         const std::vector<Asset> &assets,
                         const int std_dev_from_mean /* = 24 */)
{
    try
    {
        size_t j = 0;
        // The integration bounds are calculated by using the mean return and the standard deviation
        // of the return of each asset
        for (size_t i = 0; i < assets.size() * 2 - 1; i += 2)
        {
            integration_bounds[i]     = assets[j].getReturnMean() - std_dev_from_mean * assets[j].getReturnStdDev() + 1.0;
            integration_bounds[i + 1] = assets[j].getReturnMean() + std_dev_from_mean * assets[j].getReturnStdDev() + 1.0;
            j++;
        }
    }
    catch (const std::exception &e)
    {
        return -1;
    }
    return 0;
}


// Function to load the assets from the CSV files
// The function reads the CSV files from the directory "../data/" and populates the Asset objects
int loadAssets(const std::string &directory,
               std::vector<Asset> &assets)
{
    try
    {
        // Iterate over each entry in the directory "../data/"
        for (const auto &entry : std::filesystem::directory_iterator(directory))
        {
            // Check if the entry is a regular file and has the ".csv" extension
            if (entry.is_regular_file() && entry.path().extension() == ".csv")
            {
                Asset asset;
                // Get the filename without the extension
                std::string filename = entry.path().stem().string();
                // Set the name of the asset to the filename
                asset.setName(filename);
                // Read the CSV file and populate the asset object
                int csv_result = extrapolateCsvData(entry.path().string(), &asset);
                // Check if there was an error reading the CSV file
                if (csv_result == -1)
                {
                    std::cout << "Error reading the file " << filename << std::endl;
                    // Skip to the next file if there was an error reading the CSV file
                    continue;
                }
                // Add the asset to the vector of assets
                assets.emplace_back(asset);
            }
        }
    }
    catch (std::filesystem::filesystem_error &e)
    {
        // Handle the case where the directory couldn't be opened
        std::cout << "Could not open directory: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}


// Function called by the loadAssets function to extract the data from the CSV files
// The function reads the CSV file and calculates the mean return and the standard deviation
// of the return of the asset, and updates the Asset object
int extrapolateCsvData(const std::string &filename,
                       Asset *asset_ptr)
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        return -1;
    }

    std::string line;
    // Skip the header line
    std::getline(file, line);

    double total_return_percentage = 0.0;
    size_t counter                 = 0;
    double squared_deviation       = 0.0;
    double return_std_dev          = 0.0;
    double closing_price           = 0.0;
    double opening_price           = 0.0;
    std::vector<double> daily_returns;

    // Process each line of the file
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        // Variable to store unused fields
        std::string trash;
        // Variable to store the open price
        std::string temp_open;
        // Variable to store the close price
        std::string temp_close;

        // Extract and discard date
        std::getline(ss, trash, ',');

        // Extract and store the open price
        std::getline(ss, temp_open, ',');

        // Extract and discard high and low prices
        std::getline(ss, trash, ',');
        std::getline(ss, trash, ',');

        // Extract and store the close price
        std::getline(ss, temp_close, ',');
        opening_price = std::stod(temp_open);
        closing_price = std::stod(temp_close);
        daily_returns.emplace_back((std::stod(temp_close) - std::stod(temp_open)) / std::stod(temp_open));
        total_return_percentage += daily_returns[counter];
        counter++;
    }

    // Close the file
    file.close();

    // Calculate the mean return
    double return_mean_percentage = total_return_percentage / static_cast<double>(counter);

    // Calculate variance
    for (size_t i = 0; i < counter; i++)
    {
        squared_deviation += (daily_returns[i] - return_mean_percentage) * (daily_returns[i] - return_mean_percentage);
    }
    return_std_dev = std::sqrt(squared_deviation / static_cast<double>(counter));

    // Update the Asset object with the accumulated values
    if (asset_ptr)
    {
        asset_ptr->setReturnMean(return_mean_percentage);
        asset_ptr->setReturnStdDev(return_std_dev);
        asset_ptr->setLastRealValue(opening_price);
    }

    // Return success and print the value of the closing price of the stock
    printf("closing price of the stock is %f\n", closing_price);  
    return 0;
}