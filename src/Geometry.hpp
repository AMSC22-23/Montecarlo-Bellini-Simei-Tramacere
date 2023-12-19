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
        virtual void add_point_inside() = 0;
        virtual int get_points_inside() const = 0;

        inline int get_dimension() const { return dimension; }
        inline double get_volume() const { return volume; }
        inline double get_approximated_volume() const { return approximated_volume; }
        inline std::string get_function() const { return function; }


    protected:
        int dimension;
        double volume;
        double approximated_volume;
        std::string function;
};

#endif // GEOMETRY_HPP
