#ifndef INPUT_MANAGER_HPP
#define INPUT_MANAGER_HPP

#include <iostream>
#include <sstream>
#include <limits>
#include <vector>

void readInput(std::istream& input, std::string& value);

template <typename T>
bool parseInput(const std::string& input, T& value);

void buildIntegral(int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds);

#endif
