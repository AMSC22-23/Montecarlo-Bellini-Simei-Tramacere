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
    explicit HyperCube(int dim, double edge);

    void generateRandomPoint(std::vector<double> &random_point);

    void calculateVolume();

    int getDimension() const { return dimension; }

    double getVolume() const { return volume; }

protected: 
    double edge;
    int dimension;
    double volume;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif
