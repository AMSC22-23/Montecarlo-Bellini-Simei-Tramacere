#include <string>
#include <vector>

#include "muparser-2.3.4/include/muParser.h"

#include "Geometry.hpp"
#include "HyperSphere.hpp"
//#include "HyperRectangle.hpp"


#ifndef MC_INTEGRATOR_HPP
#define MC_INTEGRATOR_HPP

double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser);

std::pair<double, double> Montecarlo_integration(HyperSphere &domain, int n, const std::string &function, int dimension);

#endif // MC_INTEGRATOR_HPP