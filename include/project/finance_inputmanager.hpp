#ifndef PROJECT_FINANCEINPUTMANAGER_
    #define PROJECT_FINANCEINPUTMANAGER_

#include <string>
#include <vector>

#include "asset.hpp"


/**
 * @brief This function is used to get the integration bounds
 * for the Monte Carlo method
 * @details The function calculates the integration bounds
 * of the hyperrectangle domain for the Monte Carlo method
 * using the standard deviation from the mean of the assets
 * @param integration_bounds The vector that will contain the integration bounds
 * @param assets The vector of assets
 * @param std_dev_from_mean The standard deviation from the mean
 * @return 0 if the function has been executed successfully
 */
int getIntegrationBounds(std::vector<double> &integration_bounds, const std::vector<Asset> &assets, const int std_dev_from_mean = 24);

/**
 * @brief This function is used to extrapolate the data from a CSV file
 * @details The function reads the data from a CSV file and stores it in an Asset object
 * @param filename The name of the CSV file
 * @param asset_ptr The pointer to the Asset object
 * @return 0 if the function has been executed successfully
 */
int extrapolateCsvData(const std::string &filename, Asset *asset_ptr);

/**
 * @brief This function is used to load the assets from the CSV files
 * @details The function reads the data from the CSV files and stores it in a vector of Asset objects
 * @param directory The directory where the CSV files are stored
 * @param assets The vector that will contain the Asset objects
 * @return 0 if the function has been executed successfully
 */
int loadAssets(const std::string &directory, std::vector<Asset> &assets);


#endif