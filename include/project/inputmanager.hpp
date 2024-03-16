#pragma once


#include <string>
#include <vector>
#include "asset.hpp"

void input_manager( int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds);

int csv_reader(const std::string& filename, Asset* asset_ptr);