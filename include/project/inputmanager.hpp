#ifndef PROJECT_INPUTMANAGER_
    #define PROJECT_INPUTMANAGER_

#include <string>
#include <vector>
#include "asset.hpp"

void input_manager( int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds);

std::pair<std::string,std::vector<double>> create_function(int k, const std::vector<Asset>& assets);

int calculate_strike_price(const std::vector<Asset>& assets);

int set_integration_bounds(std::vector<double>& integration_bounds, const std::vector<Asset>& assets, int std_dev_from_mean = 24);

#endif