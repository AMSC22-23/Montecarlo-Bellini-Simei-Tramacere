#ifndef PROJECT_HYPERSPHERE_
    #define PROJECT_HYPERSPHERE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"

constexpr double PI = 3.14159265358979323846;

class HyperSphere: public Geometry
{
public: 
    explicit HyperSphere(size_t dim, double rad);

    void generateRandomPoint(std::vector<double> &random_point);

    inline void calculateVolume() { volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0) * std::pow(radius, dimension); }

    inline size_t getDimension() const { return dimension; }

    inline double getVolume() const { return volume; }

protected: 
    double radius;
    double parameter;
    double volume;
    size_t dimension;
    std::random_device rd;
};

#endif