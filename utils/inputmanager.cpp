#include <iostream>
#include <sstream>
#include <limits>
#include <iomanip>

#include "../include/project/inputmanager.hpp"

void buildIntegral(size_t &n,
                   size_t &dim,
                   double &rad,
                   double &edge,
                   std::string &function,
                   std::string &domain_type,
                   std::vector<double> &hyper_rectangle_bounds)
{
    std::cout << "Insert the type of domain you want to integrate:\n";
    std::cout << "  hc  - hyper-cube\n";
    std::cout << "  hs  - hyper-sphere\n";
    std::cout << "  hr  - hyper-rectangle\n";    
    readInput(std::cin, domain_type);
    while ((domain_type != "hs" && domain_type != "hr" && domain_type != "hc") || domain_type.length() == 0)
    {
        std::cout << "Invalid input. Please enter hc for hypercube, hs for hyper-sphere, hr for hyper-rectangle:\n";
        readInput(std::cin, domain_type);
    }

    std::cout << "Insert the number of random points to generate:\n";
    std::string input_n;
    readInput(std::cin, input_n);
    while (!parseInput(input_n, n) || n <= 0)
    {
        std::cout << "Invalid input. Please enter a positive integer:\n";
        readInput(std::cin, input_n);
    }

    if (domain_type == "hs")
    {
        std::cout << "Insert the dimension of the hypersphere:\n";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0)
        {
            std::cout << "Invalid input. Please enter a positive integer:\n";
            readInput(std::cin, input_dim);
        }

        std::cout << "Insert the radius of the hypersphere:\n";
        std::string input_rad;
        readInput(std::cin, input_rad);
        while (!parseInput(input_rad, rad) || rad <= 0)
        {
            std::cout << "Invalid input. Please enter a positive number:\n";
            readInput(std::cin, input_rad);
        }
    }
    else if (domain_type == "hr")
    {
        std::cout << "Insert the dimension of the hyperrectangle:\n";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0)
        {
            std::cout << "Invalid input. Please enter a positive integer:\n";
            readInput(std::cin, input_dim);
        }

        hyper_rectangle_bounds.reserve(dim * 2);
        for (size_t i = 0; i < 2 * dim; i++)
        {
            double tmp;
            size_t      current_dimension = i / 2 + 1;
            std::string boundType         = (i % 2 == 0) ? "lower" : "upper";
            std::string suffix;

              // Determine the ordinal suffix
            if (current_dimension % 10 == 1 && current_dimension % 100 != 11)
            {
                suffix = "st";
            }
            else if (current_dimension % 10 == 2 && current_dimension % 100 != 12)
            {
                suffix = "nd";
            }
            else if (current_dimension % 10 == 3 && current_dimension % 100 != 13)
            {
                suffix = "rd";
            }
            else
            {
                suffix = "th";
            }

            while (true)
            {
                std::cout << "Insert the " << boundType << " bound of the " << current_dimension
                          << suffix << " dimension of the hyper-rectangle:\n";

                std::string input_tmp;
                readInput(std::cin, input_tmp);

                if (parseInput(input_tmp, tmp))
                {
                    if (boundType == "upper" && tmp <= hyper_rectangle_bounds[i - 1])
                    {
                        std::cout << "\nInvalid input. Upper bound must be greater than lower bound.\n"
                                  << std::endl;
                    }
                    else
                    {
                        hyper_rectangle_bounds[i] = tmp;
                        break;
                    }
                }
                else
                {
                    std::cout << "\nInvalid input. Please enter a number.\n"
                              << std::endl;
                }
            }

            hyper_rectangle_bounds.emplace_back(tmp);
        }
    }
    else if (domain_type == "hc")
    {
        std::cout << "Insert the dimension of the hypercube:\n";
        std::string input_dim;
        readInput(std::cin, input_dim);
        while (!parseInput(input_dim, dim) || dim <= 0)
        {
            std::cout << "Invalid input. Please enter a positive integer:\n";
            readInput(std::cin, input_dim);
        }

        std::cout << "Insert the edge length of the hypercube:\n";
        std::string input_edge;
        readInput(std::cin, input_edge);
        while (!parseInput(input_edge, edge) || edge <= 0)
        {
            std::cout << "Invalid input. Please enter a positive number:\n";
            readInput(std::cin, input_edge);
        }
    }
    std::cout << "Insert the function to integrate:\n";
    readInput(std::cin, function);
}