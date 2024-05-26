#ifndef PROJECT_MONTECARLOINTEGRATOR_
    #define PROJECT_MONTECARLOINTEGRATOR_

#include <omp.h>
#include <iostream>

#include "hypercube.hpp"
#include "hyperrectangle.hpp"
#include "hypersphere.hpp"
#include "asset.hpp"

std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperCube &hypercube,
                                                double &variance);

std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperRectangle &hyperrectangle,
                                                double &variance);

std::pair<double, double> montecarloIntegration(size_t n,
                                                const std::string &function,
                                                HyperSphere &hypersphere,
                                                double &variance);

#endif