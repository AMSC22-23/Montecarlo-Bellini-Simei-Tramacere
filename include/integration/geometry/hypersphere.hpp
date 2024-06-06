/**
 * @file hypersphere.hpp
 * @brief This file contains the declaration of the HyperSphere class.
 */

#ifndef PROJECT_HYPERSPHERE_
    #define PROJECT_HYPERSPHERE_

#include <random>
#include <vector>
#include <cmath>
#include <omp.h>

#include "geometry.hpp" // Assuming this includes necessary headers
#include "../functionevaluator.hpp"

constexpr double PI = 3.14159265358979323846;

/**
 * @class HyperSphere
 * @brief Represents a hypersphere, a geometric shape in a space with a large number of dimensions.
 *
 * A hypersphere is a generalization of a sphere to an arbitrary number of dimensions.
 */
class HyperSphere: public Geometry
{
public: 
    /**
     * @brief Construct a new HyperSphere object
     * @param dim The dimension of the hypersphere
     * @param rad The radius of the hypersphere
     */
    explicit HyperSphere(size_t dim, double rad);

    /**
     * @brief Generate a random point inside the hypersphere
     * @details Generates a random point inside the hypersphere domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point Vector to store the random point coordinates
     */
    void generateRandomPoint(std::vector<double> &random_point) override;

    /**
     * @brief Calculate the volume of the hypersphere
     * @details The volume of a hypersphere is given by the formula: 
     * volume = (pi^parameter) / (gamma(parameter + 1)) * radius^dimension
     */
    inline void calculateVolume() override
    {
        volume = std::pow(PI, parameter) / std::tgamma(parameter + 1.0);
        for (size_t i = 0; i < dimension; ++i)
            volume *= radius;
    }

    /**
     * @brief Get the volume of the hypersphere
     * @return The volume of the hypersphere
     */
    inline double getVolume() override
    {
        return volume;
    }

    /**
     * @brief Get the dimension of the hypersphere
     * @return The dimension of the hypersphere
     */
    inline size_t getDimension() override
    {
        return dimension;
    }

private: 
    double radius;           
    double parameter;        /**< Parameter used in volume calculation */
    double volume;           
    size_t dimension;        
    std::random_device rd;  
    std::default_random_engine eng; 
};

#endif
