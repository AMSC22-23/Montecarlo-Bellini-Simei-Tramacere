#ifndef PROJECT_HYPERRECTANGLE_
#define PROJECT_HYPERRECTANGLE_

#include <random>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "geometry.hpp"
#include "functionevaluator.hpp"
#include "asset.hpp"


class HyperRectangle : public Geometry
{
public:
    explicit HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds);

    void generateRandomPoint(std::vector<double> &random_point);

    void financeGenerateRandomPoint(std::vector<double> &random_point, const std::vector<const Asset*>& assetPtrs, double std_dev_from_mean);
    
    void calculateVolume();

    int getDimension() const { return dimension; }

    double getVolume() const { return volume; }

protected:
    std::vector<double> hyper_rectangle_bounds;
    double volume;
    int dimension;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif