#ifndef PROJECT_HYPERRECTANGLE_
    #define PROJECT_HYPERRECTANGLE_

#include <random>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "geometry.hpp"
#include "functionevaluator.hpp"
#include "asset.hpp"

  /**
 * @class HyperRectangle
 * @brief This class represents a hyperrectangle, which is a geometric shape that exists in a space with a large number of dimensions.
 *
 * A hyperrectangle is a generalization of a rectangle to an arbitrary number of dimensions.
 * It is a generalization of a hypercube.
 */
class HyperRectangle: public Geometry
{
public: 
      /**
     * @brief Construct a new HyperRectangle object
     * @details Default constructor
     * @param dim An integer representing the dimension of the hyperrectangle
     * @param hyper_rectangle_bounds A vector of doubles representing the bounds of the hyperrectangle
     */
    explicit HyperRectangle(size_t dim,
                            std::vector<double> &hyper_rectangle_bounds);

      /**
     * @brief Generate a random point inside the hyperrectangle
     * @details The function generates a random point inside the hyperrectangle domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point A vector of doubles representing the random point
     */
    void generateRandomPoint(std::vector<double> &random_point) override;

      /**
     * @brief Calculate the volume of the hyperrectangle
     * @details The volume of a hyperrectangle is given by the formula: 
     * volume = (b1 - a1) * (b2 - a2) * ... * (bn - an)
     */
    inline void calculateVolume() override
    {
        for (size_t i = 0; i < 2 * dimension - 1; i += 2)
        {
            volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
        }
    }

    /**
     * @brief Get the volume of the hyperrectangle
     * @return The volume of the hyperrectangle
     */
    inline double getVolume() override
    {
        return volume;
    }

    /**
     * @brief Get the dimension of the hyperrectangle
     * @return The dimension of the hyperrectangle
     */
    inline size_t getDimension() override
    {
        return dimension;
    }

private: 
    std::vector<double> hyper_rectangle_bounds;
    double volume;
    size_t dimension;
    std::random_device rd;
    std::default_random_engine eng;
};


#endif