#ifndef PROJECT_MONTECARLOINTEGRATOR_
    #define PROJECT_MONTECARLOINTEGRATOR_

#include <omp.h>
#include <iostream>
#include "hypercube.hpp"
#include "hyperrectangle.hpp"
#include "hypersphere.hpp"
#include "asset.hpp"

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperCube &hypercube);
std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperRectangle &hyperrectangle, bool finance = false, const std::vector<const Asset*>& assetPtrs = std::vector<const Asset*>(), double std_dev_from_mean = 5.0, double* variance = nullptr);
std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperSphere &hypersphere);

#endif