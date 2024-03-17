#pragma once

#include <vector>
#include <string>


class Geometry {

public:
    Geometry() = default;
    virtual void generate_random_point(std::vector<double> &random_point) {};
    virtual void calculate_volume() {};
    //virtual std::pair<double, double> Montecarlo_integration(Geometry &integration_domain, int n, const std::string &function, int dimension) = 0;
    inline double get_volume() { return volume; };
    inline std::string get_function() const { return function; };

protected:
    int dimension;
    double volume;
    std::string function;
    
};