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
 * It is a special case of a hypercube.
 */
class HyperRectangle : public Geometry
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
    void generateRandomPoint(std::vector<double> &random_point);

    /**
     * @brief Generate a random point inside the hyperrectangle
     * @details The function generates a random point inside the hyperrectangle domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point A vector of doubles representing the random point
     * @param assetPtrs A vector of pointers to Asset objects
     * @param std_dev_from_mean A double representing the standard deviation from the mean
     * 
     * This function is used to generate random points for the finance project.
     */
    void financeGenerateRandomPoint(std::vector<double> &random_point,
                                    const std::vector<const Asset *> &assetPtrs,
                                    const double std_dev_from_mean);

    /**
     * @brief Calculate the volume of the hyperrectangle
     * @details The volume of a hyperrectangle is given by the formula:
     * volume = (b1 - a1) * (b2 - a2) * ... * (bn - an)
     */
    inline void calculateVolume()
    {
        for (size_t i = 0; i < 2 * dimension - 1; i += 2)
        {
            volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
        }
    }

    /**
     * @brief Get the dimension of the hyperrectangle
     * @return An integer representing the dimension of the hyperrectangle
     */
    inline size_t getDimension() const { return dimension; }

    /**
     * @brief Get the volume of the hyperrectangle
     * @return A double representing the volume of the hyperrectangle
     */
    inline double getVolume() const { return volume; }

protected:
    std::vector<double> hyper_rectangle_bounds;
    double volume;
    size_t dimension;
    std::random_device rd;
    std::default_random_engine eng;
};


#endif