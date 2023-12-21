#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>

#include "muParser.h"
#include "muParser.cpp"
#include "muParserBase.cpp"
#include "muParserBytecode.cpp"
#include "muParserCallback.cpp"
#include "muParserError.cpp"
#include "muParserTokenReader.cpp"

#include "project/inputmanager.hpp"
#include "project/hypersphere.hpp"
#include "project/hyperrectangle.hpp"
#include "project/hypercube.hpp"

// to compile: g++ -std=c++17 main.cpp HyperSphere.cpp input_manager.cpp mc_integrator.cpp -I{your path to muparser folder}/muparser-2.3.4/include -o main

int main(int argc, char **argv)
{
    int n, dim;
    double rad, edge;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result = {0.0, 0.0};

    // get the input from the user
    input_manager(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

    if (domain_type == "hs")
    {
        HyperSphere hypersphere(dim, rad);
        if (function == "1")
        {
            hypersphere.calculate_volume();
            result.first = hypersphere.get_volume();
        }
        else
        {
            result = hypersphere.Montecarlo_integration(n, function, dim);
        }
    }
    else if (domain_type == "hc")
    {
        HyperCube hypercube(dim, edge);
        if (function == "1")
        {
            hypercube.calculate_volume();
            result.first = hypercube.get_volume();
        }
        else
        {
            result = hypercube.Montecarlo_integration(n, function);
        }
    }
    else if (domain_type == "hr")
    {
        HyperRectangle hyperrectangle(dim, hyper_rectangle_bounds);
        if (function == "1")
        {
            hyperrectangle.calculate_volume();
            result.first = hyperrectangle.get_volume();
        }
        else
        {
            result = hyperrectangle.Montecarlo_integration(n, function, dim);
        }
    }
    else
    {
        std::cout << "Invalid input.";
    }
    std::cout << std::endl
              << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    if (result.second != 0.0) 
        std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;

    // calculate the exact domain if the user tries to integrate the volume of the hypersphere
    // TODO with various domains
    /*if (function == "1")
    {
        integration_domain.calculate_volume();
        double exact_domain = hypersphere.get_volume();
        std::cout << "The exact domain in " << dim << " dimensions you were looking for is: " << exact_domain << std::endl;
        std::cout << "The absolute error is: " << std::abs(result.first - exact_domain) << std::endl;
        std::cout << "The relative error is: " << std::abs(result.first - exact_domain) / exact_domain << std::endl;
    }*/
    return 0;
}
