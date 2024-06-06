#include "../../include/optionpricing/finance_inputmanager.hpp"

double logReturn(const double price, const double previous_price)
{
    return std::log(price/previous_price);
}

  // Function to calculate Hyperrectangle integration bounds
int setIntegrationBounds(std::vector<double> &integration_bounds,
                         const std::vector<Asset> &assets,
                         const int std_dev_from_mean /* = 24 */)
{
    try
    {
        size_t j = 0;
        for (size_t i = 0; i < assets.size() * 2 - 1; i += 2)
        {
              // Calculate the integration bounds for each asset
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

// Function to load assets from CSV files
LoadAssetError loadAssets(const std::string& directory, std::vector<Asset>& assets, const AssetCountType& asset_count_type) {
    std::string subdirectory;

    // Determine subdirectory based on option type
    switch (asset_count_type) {
        case AssetCountType::Single:
            subdirectory = "single_asset";
            break;
        case AssetCountType::Multiple:
            subdirectory = "multi_asset";
            break;
        default:
            std::cerr << "Invalid option type" << std::endl;
            return LoadAssetError::DirectoryOpenError;
    }

    std::filesystem::path targetDirectory = std::filesystem::path(directory) / subdirectory;

    try {
        bool validFileFound = false;

        // Iterate over each entry in the subdirectory
        for (const auto& entry : std::filesystem::directory_iterator(targetDirectory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                validFileFound = true;
                Asset asset;
                std::string filename = entry.path().stem().string();
                asset.setName(filename);
                int csv_result = extrapolateCsvData(entry.path().string(), &asset);

                if (csv_result == -1) {
                    std::cout << "Error reading the file " << filename << std::endl;
                    return LoadAssetError::FileReadError;
                }

                assets.emplace_back(asset);
            }
        }

        if (!validFileFound) {
            return LoadAssetError::NoValidFiles;
        }
    } catch (std::filesystem::filesystem_error& e) {
        std::cerr << "Could not open directory: " << e.what() << std::endl;
        return LoadAssetError::DirectoryOpenError;
    }

    return LoadAssetError::Success;
}
  // Function to extrapolate data from a CSV file
int extrapolateCsvData(const std::string &filename,
                       Asset *asset_ptr)
{
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
    double std_dev                 = 0.0;
    double closing_price           = 0.0;
    std::vector<double> daily_returns;

      // Process each line of the file
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string trash;       // Variable to store unused fields
        std::string temp_open;   // Variable to store the open price
        std::string temp_close;  // Variable to store the close price

          // Extract and discard date
        std::getline(ss, trash, ',');

          // Extract and store the open price
        std::getline(ss, temp_open, ',');

          // Extract and discard high and low prices
        std::getline(ss, trash, ',');
        std::getline(ss, trash, ',');

          // Extract and store the close price
        std::getline(ss, temp_close, ',');
        closing_price = std::stod(temp_close);
          // daily_returns.emplace_back((std::stod(temp_close) - std::stod(temp_open)) / std::stod(temp_open));
        daily_returns.emplace_back(logReturn(std::stod(temp_close), std::stod(temp_open)));
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
        std_dev += (daily_returns[i] - return_mean_percentage) * (daily_returns[i] - return_mean_percentage) / static_cast<double>(counter);
    }
    std_dev = std::sqrt(std_dev);

      // Update the Asset object with the accumulated values
    if (asset_ptr)
    {
        asset_ptr->setReturnMean(return_mean_percentage);
        asset_ptr->setReturnStdDev(std_dev);
        asset_ptr->setLastRealValue(closing_price);
        asset_ptr->setDailyReturns(daily_returns);
    }

    return 0;
}