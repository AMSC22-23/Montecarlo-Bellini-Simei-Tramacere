#ifndef PROJECT_FINANCEINPUTMANAGER_
    #define PROJECT_FINANCEINPUTMANAGER_

#include <string>
#include <vector>

#include "asset.hpp"

int getIntegrationBounds(std::vector<double> &integration_bounds, const std::vector<Asset> &assets, int std_dev_from_mean = 24);

int extrapolateCsvData(const std::string &filename, Asset *asset_ptr);

int loadAssets(const std::string &directory, std::vector<Asset> &assets);

#endif