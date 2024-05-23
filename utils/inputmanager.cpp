#include "../include/project/inputmanager.hpp"
#include <iostream>
#include <sstream>
#include <limits>
#include <iomanip>


void readInput(std::istream& input, std::string& value) {
    input >> std::ws; // Skip whitespace
    std::getline(input, value);
}

template <typename T>
bool parseInput(const std::string& input, T& value) {
    try {
        value = std::stod(input);
        return true;
    } catch (...) {
        return false;
    }
}


void buildIntegral(int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds) {
    std::cout << "Insert the type of domain you want to integrate: (hc for hyper-cube, hs for hyper-sphere, hr for hyper-rectangle)" << std::endl;
    readInput(std::cin, domain_type);
    while ((domain_type != "hs" && domain_type != "hr" && domain_type != "hc") || domain_type.length() == 0) {
        std::cout << "Invalid input. Please enter hc for hypercube, hs for hyper-sphere, hr for hyper-rectangle: ";
        readInput(std::cin, domain_type);
    }

    std::cout << "Insert the number of random points to generate: ";
    std::string input_n;
    readInput(std::cin, input_n);
    while (!parseInput(input_n, n) || n <= 0) {
        std::cout << "Invalid input. Please enter a positive integer: ";
        readInput(std::cin, input_n);
    }

    if (domain_type == "hs") {
        std::cout << "Insert the dimension of the hypersphere: ";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0) {
            std::cout << "Invalid input. Please enter a positive integer: ";
            readInput(std::cin, input_dim);
        }

        std::cout << "Insert the radius of the hypersphere: ";
        std::string input_rad;
        readInput(std::cin, input_rad);
        while (!parseInput(input_rad, rad) || rad <= 0) {
            std::cout << "Invalid input. Please enter a positive number: ";
            readInput(std::cin, input_rad);
        }
    } else if (domain_type == "hr") {
        std::cout << "Insert the dimension of the hyperrectangle: ";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0) {
            std::cout << "Invalid input. Please enter a positive integer: ";
            readInput(std::cin, input_dim);
        }

        hyper_rectangle_bounds.reserve(dim * 2);
        for (int i = 0; i < 2 * dim; i++) {
            double tmp;
            if (i == 0)
                std::cout << "Insert the 1st dimension coordinate of the hyper-rectangle: ";
            else if (i == 1)
                std::cout << "Insert the 2nd dimension coordinate of the hyper-rectangle: ";
            else if (i == 2)
                std::cout << "Insert the 3rd dimension coordinate of the hyper-rectangle: ";
            else
                std::cout << "Insert the " << i+1 << "th dimension coordinate of the hyper-rectangle: ";

            std::string input_tmp;
            readInput(std::cin, input_tmp);
            while (!parseInput(input_tmp, tmp)) {
                std::cout << "Invalid input. Please enter a number: ";
                readInput(std::cin, input_tmp);
            }
            hyper_rectangle_bounds.push_back(tmp);
        }
    } else if (domain_type == "hc") {
        std::cout << "Insert the dimension of the hypercube: ";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0) {
            std::cout << "Invalid input. Please enter a positive integer: ";
            readInput(std::cin, input_dim);
        }

        std::cout << "Insert the edge length of the hypercube: ";
        std::string input_edge;
        readInput(std::cin, input_edge);
        while (!parseInput(input_edge, edge) || edge <= 0) {
            std::cout << "Invalid input. Please enter a positive number: ";
            readInput(std::cin, input_edge);
        }
    }

    std::cout << "Insert the function to integrate: ";
    readInput(std::cin, function);
}
