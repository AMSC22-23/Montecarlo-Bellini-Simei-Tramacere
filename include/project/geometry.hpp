#ifndef PROJECT_GEOMETRY_
#define PROJECT_GEOMETRY_

#include <vector>
#include <string>
#include "asset.hpp" // Include the header for the Asset class

class Geometry
{
public:
    Geometry() = default;

    virtual void generateRandomPoint(std::vector<double> &random_point) = 0;
    virtual void calculateVolume() = 0;
    //virtual std::pair<double, double> Montecarlo_integration(Geometry &integration_domain, int n, const std::string &function, int dimension) = 0;

    inline double getVolume() const { return volume; }

protected:
    int dimension;
    double volume;
};

#endif