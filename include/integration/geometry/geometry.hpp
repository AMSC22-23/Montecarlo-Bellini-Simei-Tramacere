/**
 * @file geometry.hpp
 * @brief This file contains the declaration of the Geometry class.
 */

#ifndef PROJECT_GEOMETRY_
    #define PROJECT_GEOMETRY_

#include <vector>
#include <string>
#include <iostream>

#include "../../optionpricing/asset.hpp"

/**
 * @class Geometry
 * @brief Represents a geometry.
 *
 * This abstract class is used to represent a geometry,
 * and the derived classes are used to represent different
 * geometries such as a hyperrectangle, a hypersphere, and a hypercube.
 */
class Geometry
{
public: 
    /**
     * @brief Generate a random point inside the geometry
     * @details Generates a random point inside the geometry domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point Vector to store the random point coordinates
     */
    virtual void generateRandomPoint(std::vector<double> &random_point) = 0;

    /**
     * @brief Calculate the volume of the geometry
     */
    virtual inline void calculateVolume() = 0;

    /**
     * @brief Get the volume of the geometry
     * @return The volume of the geometry
     */
    virtual inline double getVolume() = 0;

    /**
     * @brief Get the dimension of the geometry
     * @return The dimension of the geometry
     */
    virtual inline size_t getDimension() = 0;

    /**
     * @brief Destructor
     */
    virtual ~Geometry() {}

protected: 
    size_t dimension;
    double volume;
};

#endif