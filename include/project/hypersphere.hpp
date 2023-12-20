#ifndef PROJECT_HYPERSPHERE_
    #define PROJECT_HYPERSPHERE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"


class HyperSphere : public Geometry {
    
    public:
        HyperSphere(int dim, double rad);

        void generate_random_point(std::vector<double> &random_point);
        void calculate_volume();
        void calculate_approximated_volume(int n);
        void add_point_inside();

        int get_points_inside() const;
        int get_dimension() const;

    protected:
        double radius;
        double parameter;
        double hypercube_volume;
        int points_inside;
        std::random_device rd;
        std::default_random_engine eng;
};


#endif