#ifndef PROJECT_GEOMETRY_
#define PROJECT_GEOMETRY_

#include <vector>
#include <string>
#include <iostream>
#include "asset.hpp"  // Include the header for the Asset class

  /**
 * @brief This class is used to represent a geometry
 * @details The abstract class is used to represent a geometry,
 * and the derived classes are used to represent different
 * geometries such as a hyperrectangle, a hypersphere, and a hypercube
 */
class Geometry
{
public: 
      /**
     * @brief Generate a random point inside the geometry
     * @details The function generates a random point inside the geometry domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point A vector of doubles representing the random point
     */
    virtual void generateRandomPoint(std::vector<double> &random_point) = 0;

      /**
     * @brief Calculate the volume of the geometry
     * @details The function calculates the volume of the geometry
     */
    virtual inline void calculateVolume() = 0;

      /**
     * @brief Get the volume of the geometry
     * @return An integer representing the volume of the geometry
     */
    virtual inline double getVolume() = 0;

      /**
     * @brief Get the dimension of the geometry
     * @return An integer representing the dimension of the geometry
     */
    virtual inline size_t getDimension() = 0;

      /**
     * @brief Destroy the Geometry object
     * @details Destructor
     */
    virtual ~Geometry() {}

protected: 
    size_t dimension;
    double volume;
};

#endif