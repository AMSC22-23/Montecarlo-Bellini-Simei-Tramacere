#include "project/inputmanager.hpp"
#include <iostream>
#include <chrono>
#include <vector>

void input_manager(int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds)
{
    // Get and validate number of random points
    std::cout << "Insert the type of domain you want to integrate: (hc for hyper-cube, hs for hyper-sphere, hr for hyper-rectangle)" << std::endl;
    std::cin >> domain_type;
    while (std::cin.fail() || (domain_type != "hs" && domain_type != "hr" && domain_type != "hc"))
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

    if (domain_type == "hs")
    {
        std::cout << "Insert the dimension of the hypersphere: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }

        std::cout << "Insert the radius of the hypersphere: ";
        std::cin >> rad;
        while (std::cin.fail() || rad <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> rad;
        }
    }
    else if (domain_type == "hr")
    {
        std::cout << "Insert the dimension of the hyperrectangle: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }

        hyper_rectangle_bounds.reserve(dim*2);
        double tmp;

        for (int i = 0; i < 2 * dim; i++)
        {
            if (i == 0)
                std::cout << "Insert the 1st dimension coordinate of the hyper-rectangle: ";
            else if (i == 1)
                std::cout << "Insert the 2nd dimension coordinate of the hyper-rectangle: ";
            else if (i == 2)
                std::cout << "Insert the 3rd dimension coordinate of the hyper-rectangle: ";
            else
                std::cout << "Insert the " << i+1 << "th dimension coordinate of the hyper-rectangle: ";

            std::cin >> tmp;
            hyper_rectangle_bounds.push_back(tmp);
            while (std::cin.fail())
            {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a positive number: ";
                std::cin >> hyper_rectangle_bounds[i];
            }
        }
    }
    else if (domain_type == "hc")
    {
        std::cout << "Insert the dimension of the hypercube: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }
        std::cout << "Insert the edge length of the hypercube: ";
        std::cin >> edge;
        while (std::cin.fail() || edge <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> rad;
        }
    }

    // ask the user to insert the function to integrate
    std::cout << "Insert the function to integrate: ";
    std::cin >> function;
}