#ifndef INPUT_MANAGER_HPP
    #define INPUT_MANAGER_HPP

#include <iostream>
#include <sstream>
#include <limits>
#include <vector>

inline void readInput(std::istream &input, std::string &value)
{
    input >> std::ws;  // Skip whitespace
    std::getline(input, value);
}

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

void buildIntegral(size_t &n, size_t &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds);

#endif
