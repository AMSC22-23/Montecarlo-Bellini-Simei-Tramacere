#ifndef PROJECT_HYPERCUBE_
    #define PROJECT_HYPERCUBE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


class HyperCube : public Geometry
{
public:
    HyperCube(int dim, double &edge);

    void generate_random_point(std::vector<double> &random_point);

    void calculate_volume();

    std::pair<double, double> Montecarlo_integration(int n, const std::string &function);

protected:
    double edge;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif