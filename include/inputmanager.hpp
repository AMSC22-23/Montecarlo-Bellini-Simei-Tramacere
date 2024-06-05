#ifndef INPUT_MANAGER_HPP
    #define INPUT_MANAGER_HPP

#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <functional>
#include <string>

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
    std::istringstream iss(input);
    return (iss >> value) ? true: false;
}

  /**
 * @brief Template function to read and validate input
 * @details This function reads an input from the user, parses it, and validates it using a custom validator.
 * @param prompt A string containing the prompt to display to the user
 * @param value A reference to the variable where the parsed value will be stored
 * @param validator A function that takes a reference to the parsed value and returns true if it is valid
 * @tparam T The type of the value to be read and validated
 */
template <typename T>
void readValidatedInput(const std::string &prompt, T &value, std::function<bool(const T &)> validator)
{
    std::string input;
    while (true)
    {
        std::cout << prompt;
        readInput(std::cin, input);
        if (parseInput(input, value) && validator(value))
            break;
        std::cout << "Invalid input. Please try again:\n";
    }
}

  /**
 * @brief Calculate the approximated value of the integral over a chosen domain
 * @details This function reads the input from the user and builds the integral over a chosen domain using the Monte Carlo method.
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

#endif  // INPUT_MANAGER_HPP
