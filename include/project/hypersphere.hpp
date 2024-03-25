#ifndef PROJECT_HYPERSPHERE_
    #define PROJECT_HYPERSPHERE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"



class HyperSphere : public Geometry
{
public:
    HyperSphere(int dim, double rad);

    void generate_random_point(std::vector<double> &random_point);

    std::pair<double, double> Montecarlo_integration(int n, const std::string &function, int dimension);

    void calculate_volume();

    void add_point_inside();

    int getdimension();

    int get_points_inside() const;

protected:
    double radius;
    double parameter;
    int points_inside;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif