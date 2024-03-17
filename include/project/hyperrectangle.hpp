#pragma once

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


class HyperRectangle : public Geometry {

public:
    HyperRectangle(const HyperRectangle &other);
    HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds);
    void generate_random_point(std::vector<double> &random_point);
    void calculate_volume();

protected:
    std::vector<double> hyper_rectangle_bounds;
    std::random_device rd;
    std::default_random_engine eng;
    
};