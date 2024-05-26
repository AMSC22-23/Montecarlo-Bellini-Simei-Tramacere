#ifndef PROJECT_HYPERCUBE_
    #define PROJECT_HYPERCUBE_

#include <random>
#include <vector>
#include <cmath>

#include "geometry.hpp"
#include "functionevaluator.hpp"


/**
 * @class HyperCube
 * @brief This class represents a hypercube, which is a geometric shape that exists in a space with a large number of dimensions.
 * 
 * A hypercube is a generalization of a cube to an arbitrary number of dimensions.
 * It is a special case of a hyperrectangle.
 */
class HyperCube: public Geometry
{
public: 
    /**
     * @brief Construct a new HyperCube object
     * @details Default constructor
     * @param dim An integer representing the dimension of the hypercube
     * @param edge A double representing the edge of the hypercube
     */
    explicit HyperCube(size_t dim, double edge);

    /**
     * @brief Generate a random point inside the hypercube
     * @details The function generates a random point inside the hypercube domain
     * in a parallel fashion using OpenMP following a uniform distribution.
     * @param random_point A vector of doubles representing the random point
     */
    void generateRandomPoint(std::vector<double> &random_point);

    /**
     * @brief Calculate the volume of the hypercube
     * 
     * The volume of a hypercube is given by the formula:
     * volume = edge^dimension
     */
    inline void calculateVolume()
    {
        for (size_t i = 0; i < dimension; ++i)
            volume *= edge;
    }

    /**
     * @brief Get the dimension of the hypercube
     * @return An integer representing the dimension of the hypercube
     */
    inline size_t getDimension() const { return dimension; }

    /**
     * @brief Get the volume of the hypercube
     * @return A double representing the volume of the hypercube
     */
    inline double getVolume() const { return volume; }

protected: 
    double edge;
    size_t dimension;
    double volume;
    std::random_device rd;
    std::default_random_engine eng;
};


#endif