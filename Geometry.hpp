#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <vector>
#include <string>

class Geometry
{
    public:
        Geometry() = default;
        
        virtual void generate_random_point(std::vector<double>& random_point) = 0;

        virtual void calculate_volume() = 0;
        virtual void calculate_approximated_volume(int) = 0;

        int get_dimension() const { return dimension; }
        double get_volume() const { return volume; }
        double get_approximated_volume() const { return approximated_volume; }
        std::string get_function() const { return function; }

    protected:
        int dimension;
        double volume;
        double approximated_volume;
        std::string function;
};

#endif // GEOMETRY_HPP
