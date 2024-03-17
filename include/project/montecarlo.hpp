#pragma once

#include <utility>
#include <string>

#include "hypersphere.hpp"
#include "hyperrectangle.hpp"
#include "hypercube.hpp"


std::pair<double, double> hs_montecarlo_integration(HyperSphere hs, int n, const std::string &function, int dimension);

std::pair<double, double> hc_montecarlo_integration(HyperCube hc, int n, const std::string &function, int dimension);

std::pair<double, double> hr_montecarlo_integration(HyperRectangle hr, int n, const std::string &function, int dimension);