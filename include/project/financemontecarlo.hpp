#ifndef PROJECT_FINANCEMONTECARLO_HPP
    #define PROJECT_FINANCEMONTECARLO_HPP

#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "hyperrectangle.hpp"
#include "asset.hpp"

std::pair<double, double> montecarloPricePrediction(int points, const std::string &function,
                                                    HyperRectangle &hyperrectangle,
                                                    const std::vector<const Asset *> &assetPtrs,
                                                    double std_dev_from_mean,
                                                    double &variance,
                                                    std::vector<double> coefficients,
                                                    int strike_price);

#endif