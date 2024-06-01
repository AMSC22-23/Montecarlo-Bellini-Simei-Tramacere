#ifndef INPUT_MANAGER_HPP
    #define INPUT_MANAGER_HPP

#include <iostream>
#include <sstream>
#include <limits>
#include <vector>

  /**
 * @brief Read an input from the user
 * @details This function reads an input from the user and returns it as a string.
 * @param input A reference to an input stream
 * @param value A reference to a string
 */
inline void readInput(std::istream &input, std::string &value)
{
      // Skip whitespace
    input >> std::ws;
    std::getline(input, value);
}

  /**
 * @brief Parse an input
 * @details This function tries to parse a string into a value of type T.
 * @param input A reference to a string
 * @tparam value A reference to a value of type T
 * @return True if the parsing was successful, false otherwise
 */
template <typename T>
inline bool parseInput(const std::string &input, T &value)
{
    try
    {
        value = std::stod(input);
        return true;
    }
    catch (...)
    {
        return false;
    };
}

  /**
 * @brief Calculate the approssimated value of the integral over a chosen domain
 * @details This function reads the input from the user and builds the integral
 * over a chosen domain using the Monte Carlo method.
 * @param n A reference to the number of points to generate
 * @param dim A reference to the dimension of the domain
 * @param rad A reference to the radius of the hypersphere
 * @param edge A reference to the edge of the hypercube
 * @param function A reference to the function to integrate
 * @param domain_type A reference to the domain type
 * @param hyper_rectangle_bounds A reference to the bounds of the hyperrectangle
 *
 * The function is the core of the original code, which is used to build the integral
 * over a chosen domain using the Monte Carlo method. The function reads the input from
 * the user and builds the integral over a chosen domain using the Monte Carlo method.
 */
void buildIntegral(size_t &n, size_t &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds);

#endif