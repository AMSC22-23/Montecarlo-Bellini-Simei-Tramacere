#ifndef PROJECT_MONTECARLO_
    #define PROJECT_MONTECARLO_

#include <string>
#include <vector>

#include "muParser.h"
#include "geometry.hpp"
#include "hypersphere.hpp"
//#include "hyperrectangle.hpp"


double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser);

std::pair<double, double> Montecarlo_integration(HyperSphere &domain, int n, const std::string &function, int dimension);


#endif