#include <random>
#include <vector>
#include <cmath>

#include "Geometry.hpp"


class HyperSphere: public Geometry
{
    public:
        HyperSphere(int dim, double rad) {
            dimension = dim;
            radius = rad;
            parameter = dim / 2.0;
            hypercube_volume = pow(2 * radius, dimension);
            points_inside = 0;
        }

        std::vector<double> generate_random_point() override {
            // setup the random number generator
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<double> distribution(-radius, radius);
            // create a vector to store the random point
            std::vector<double> random_point;
            random_point.reserve(dimension);
            // generate the random point     
            for (int i = 0; i < dimension; ++i) {                                
                random_point.push_back(distribution(eng));
            }
            // check if the point is inside the hypersphere
            double sum_of_squares = std::accumulate(random_point.begin(), random_point.end(), 
                    0.0, [](double sum, double x) { return sum + std::pow(x, 2); });
            if (sum_of_squares <= std::pow(radius, 2)) return random_point;
            else return std::vector<double>();
        }

        void calculate_volume() override {
            volume = pow(M_PI, parameter) / tgamma(parameter + 1.0) * pow(radius, dimension);
        }

        void calculate_approximated_volume(int n) override {
            approximated_volume = (static_cast<double>(points_inside) / n) * hypercube_volume;
        }

        void add_point_inside() { ++points_inside; }

    protected:
        double radius;
        double parameter;
        double hypercube_volume;
        int points_inside;
};