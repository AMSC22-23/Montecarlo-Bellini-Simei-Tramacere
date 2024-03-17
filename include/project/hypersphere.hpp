#pragma once

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


class HyperSphere : public Geometry {

public:
    HyperSphere(const HyperSphere &other);
    HyperSphere(int dim, double rad);
    void generate_random_point(std::vector<double> &random_point);
    void calculate_volume();
    void add_point_inside();
    int get_points_inside() const;

protected:
    double radius;
    double parameter;
    int points_inside;
    std::random_device rd;
    std::default_random_engine eng;
    
};