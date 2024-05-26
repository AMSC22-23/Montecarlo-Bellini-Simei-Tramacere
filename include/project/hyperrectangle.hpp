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
    explicit HyperRectangle(size_t dim,
                            std::vector<double> &hyper_rectangle_bounds);

    void generateRandomPoint(std::vector<double> &random_point);

    void financeGenerateRandomPoint(std::vector<double> &random_point,
                                    const std::vector<const Asset *> &assetPtrs,
                                    const double std_dev_from_mean);

    inline void calculateVolume()
    {
        for (size_t i = 0; i < 2 * dimension - 1; i += 2)
        {
            volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
        }
    }

    inline size_t getDimension() const { return dimension; }

    inline double getVolume() const { return volume; }

protected:
    std::vector<double> hyper_rectangle_bounds;
    double volume;
    size_t dimension;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif