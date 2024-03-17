#pragma once

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


class HyperCube : public Geometry {

public:
    HyperCube(const HyperCube &other);
    HyperCube(int dim, double &edge);
    void generate_random_point(std::vector<double> &random_point);
    void calculate_volume();

protected:
    double edge;
    std::random_device rd;
    std::default_random_engine eng;
    
};