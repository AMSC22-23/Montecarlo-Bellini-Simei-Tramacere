#include "input_manager.hpp"
#include <iostream>
#include <chrono>

void input_manager( int &n, int &dim, double &rad, std::string &function, std::string &domain_type)
{
    // Get and validate number of random points
    std::cout << "Insert the type of domain you want to integrate: (hs for hyper-sphere, hr for hyper-rectangle)" << std::endl;
    std::cin >> domain_type;
    while (std::cin.fail() || (domain_type != "hs" && domain_type != "hr"))
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter hs for hyper-sphere, hr for hyper-rectangle: ";
        std::cin >> domain_type;
    }
    std::cout << "Insert the number of random points to generate: ";
    std::cin >> n;
    while (std::cin.fail() || n <= 0)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter a positive integer: ";
        std::cin >> n;
    }

    // Get and validate dimension of hypersphere
    std::cout << "Insert the dimension of the hypersphere: ";
    std::cin >> dim;
    while (std::cin.fail() || dim <= 0)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter a positive integer: ";
        std::cin >> dim;
    }

    // Get and validate radius of hypersphere
    std::cout << "Insert the radius of the hypersphere: ";
    std::cin >> rad;
    while (std::cin.fail() || rad <= 0)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter a positive number: ";
        std::cin >> rad;
    }
    // ask the user to insert the function to integrate
    std::cout << "Insert the function to integrate: ";
    std::cin >> function;
}