#ifndef PROJECT_HYPERRECTANGLE_
#define PROJECT_HYPERRECTANGLE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


class HyperRectangle : public Geometry
{
public:
    HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds);

    void generate_random_point(std::vector<double> &random_point);

    void calculate_volume();

    void calculate_approximated_volume(int n);

    void add_point_inside();

    std::pair<double, double> Montecarlo_integration(int n, const std::string &function, int dimension);

    int get_points_inside() const;

    int get_dimension() const;

protected:
    double radius;
    double parameter;
    double hypercube_volume;
    int points_inside;
    std::vector<double> hyper_rectangle_bounds;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif