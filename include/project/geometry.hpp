    #ifndef PROJECT_GEOMETRY_
#define PROJECT_GEOMETRY_

#include <vector>
#include <string>
#include "asset.hpp" // Include the header for the Asset class


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
     * @brief Construct a new Geometry object
     * @details Default constructor
     */
    Geometry() = default;

    /**
     * @brief Construct a new Geometry object
     * @details Custom constructor
     * @param dimension Dimension of the geometry
     */
    virtual void generateRandomPoint(std::vector<double> &random_point) = 0;

    /**
     * @brief Calculate the volume of the geometry
     * @details The function calculates the volume of the geometry
     */
    virtual void calculateVolume() = 0;

    /**
     * @brief Get the volume of the geometry
     * @return An integer representing the volume of the geometry
     */
    inline double getVolume() const { return volume; }

protected:
    int dimension;
    double volume;
};


#endif