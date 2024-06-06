/**
 * @file hypercube.hpp
 * @brief This file contains the declaration of the HyperCube class.
 */

#ifndef PROJECT_HYPERCUBE_
    #define PROJECT_HYPERCUBE_

#include <random>
#include <vector>
#include <cmath>
#include <omp.h>

#include "geometry.hpp"
#include "../functionevaluator.hpp"

/**
 * @class HyperCube
 * @brief Represents a hypercube, a geometric shape in a space with a large number of dimensions.
 *
 * A hypercube is a generalization of a cube to an arbitrary number of dimensions.
 * It is a special case of a hyperrectangle.
 */
class HyperCube: public Geometry
{
public: 
    /**
     * @brief Construct a new HyperCube object
     * @param dim The dimension of the hypercube
     * @param edge The edge length of the hypercube
     */
    explicit HyperCube(size_t dim, double edge);

    /**
     * @brief Generate a random point inside the hypercube
     * @details Generates a random point inside the hypercube domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point Vector to store the random point coordinates
     */
    void generateRandomPoint(std::vector<double> &random_point) override;

    /**
     * @brief Calculate the volume of the hypercube
     *
     * The volume of a hypercube is given by the formula: 
     * volume = edge^dimension
     */
    inline void calculateVolume() override
    {
        for (size_t i = 0; i < dimension; ++i)
            volume *= edge;
    }

    /**
     * @brief Get the volume of the hypercube
     * @return The volume of the hypercube
     */
    inline double getVolume() override
    {
        return volume;
    }

    /**
     * @brief Get the dimension of the hypercube
     * @return An integer representing the dimension of the hypercube
     */
    inline size_t getDimension() override
    {
        return dimension;
    }

private: 
    double edge;
    size_t dimension;
    double volume;
    std::random_device rd;
    std::default_random_engine eng;
};

#endif