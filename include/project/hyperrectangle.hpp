#ifndef PROJECT_HYPERRECTANGLE_
#define PROJECT_HYPERRECTANGLE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"
#include "asset.hpp"


class HyperRectangle : public Geometry
{
public:
    HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds);

    void generate_random_point(std::vector<double> &random_point, bool finance = false, const std::vector<const Asset*>& assetPtrs = std::vector<const Asset*>(), double std_dev_from_mean = 5.0);

    void calculate_volume();

    int getdimension();

protected:
    std::vector<double> hyper_rectangle_bounds;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif