#ifndef PROJECT_HYPERSPHERE_
    #define PROJECT_HYPERSPHERE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"

class HyperSphere: public Geometry
{
public: 
    explicit HyperSphere(int dim, double rad);

    void generateRandomPoint(std::vector<double> &random_point);

    void calculateVolume();

    int getDimension() const { return dimension; }

    double getVolume() const { return volume; }

protected: 
    double radius;
    double parameter;
    double volume;
    int dimension;
    std::random_device rd;
};

#endif