/**
 * @file finance_inputmanager.hpp
 * @brief This file contains the declarations of functions related to input management for the pricing of an option.
 
 */

#ifndef PROJECT_FINANCEINPUTMANAGER_
    #define PROJECT_FINANCEINPUTMANAGER_

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "asset.hpp"
#include "optionpricer.hpp"
#include "finance_enums.hpp"

  /**
 * @brief Calculate the log return of an asset.
 * @details This function calculates the log return, which is the natural logarithm of the ratio of the current price to the previous price.
 * @param price The current price of the asset.
 * @param previous_price The previous price of the asset.
 * @return The log return.
 */
double logReturn(const double price, const double previous_price);

  /**
 * @brief Extrapolate data from a CSV file.
 * @details This function reads the data from a CSV file and stores it in an Asset object.
 * @param filename The name of the CSV file.
 * @param asset_ptr The pointer to the Asset object.
 * @return 0 if the function has been executed successfully.
 */
int extrapolateCsvData(const std::string &filename, Asset *asset_ptr);

  /**
 * @brief Load assets from CSV files.
 * @details This function reads the data from CSV files in a specified directory
 *          and stores it in a vector of Asset objects.
 * @param directory The directory where the CSV files are stored.
 * @param assets The vector that will contain the Asset objects.
 * @return A LoadAssetError value that indicates the status of the function.
 */
LoadAssetError loadAssets(const std::string &directory, std::vector<Asset> &assets, const AssetCountType &asset_count_type);

#endif
