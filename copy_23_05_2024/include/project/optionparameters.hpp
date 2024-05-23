#ifndef OPTION_PARAMETERS_
    #define OPTION_PARAMETERS_

#include <vector>
#include <string>
#include <cmath>

#include "asset.hpp"

std::pair<std::string, std::vector<double>> createPayoffFunction(double k, const std::vector<Asset> &assets);

double calculateStrikePrice(const std::vector<Asset> &assets);

#endif