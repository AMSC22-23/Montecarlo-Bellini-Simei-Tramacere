#ifndef PROJECT_MONTECARLOINTEGRATOR_
    #define PROJECT_MONTECARLOINTEGRATOR_

#include <omp.h>
#include <iostream>
#include "hypercube.hpp"
#include "hyperrectangle.hpp"
#include "hypersphere.hpp"

    std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperCube &hypercube);
    std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperRectangle &hyperrectangle);
    std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperSphere &hypersphere);

#endif