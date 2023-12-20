#ifndef PROJECT_GEOMETRY_
    #define PROJECT_GEOMETRY_
    
#include <vector>
#include <string>

class Geometry
{
public:
    Geometry() = default;

    virtual void generate_random_point(std::vector<double> &random_point) = 0;
    virtual void calculate_volume() = 0;
    //virtual std::pair<double, double> Montecarlo_integration(Geometry &integration_domain, int n, const std::string &function, int dimension) = 0;

    inline double get_volume() const { return volume; }
    inline std::string get_function() const { return function; }

protected:
    int dimension;
    double volume;
    std::string function;
};

#endif