#ifndef PROJECT_HYPERCUBE_
    #define PROJECT_HYPERCUBE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"

class HyperCube: public Geometry
{
public: 
    explicit HyperCube(size_t dim, double edge);

    void generateRandomPoint(std::vector<double> &random_point);

    inline void calculateVolume()
    {
        for (size_t i = 0; i < dimension; ++i)
            volume *= edge;
    }

    inline size_t getDimension() const { return dimension; }

    inline double getVolume() const { return volume; }

protected: 
    double edge;
    size_t dimension;
    double volume;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif
