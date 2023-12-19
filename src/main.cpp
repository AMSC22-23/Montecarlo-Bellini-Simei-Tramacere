#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>

#include "muparser-2.3.4/include/muParser.h"
#include "muparser-2.3.4/src/muParser.cpp"
#include "muparser-2.3.4/src/muParserBase.cpp"
#include "muparser-2.3.4/src/muParserBytecode.cpp"
#include "muparser-2.3.4/src/muParserCallback.cpp"
#include "muparser-2.3.4/src/muParserError.cpp"
#include "muparser-2.3.4/src/muParserTokenReader.cpp"

#include "input_manager.hpp"
#include "MonteCarlo.hpp"

// to compile: g++ -std=c++17 main.cpp HyperSphere.cpp input_manager.cpp mc_integrator.cpp -I{your path to muparser folder}/muparser-2.3.4/include -o main

int main(int argc, char **argv)
{
    int n, dim;
    double rad;
    std::string function;
    std::string domain_type;

    // get the input from the user
    input_manager(n, dim, rad, function, domain_type);

    //TODO
    /*if (domain_type == "hs")
    {
        HyperSphere hypersphere(dim, rad);
        std::pair<double, double> result = Montecarlo_integration(hypersphere, n, function, dim);
    }
    else if (domain_type == "hr")
    {
        // HyperRectangle hyperrectangle(dim, rad);
        // std::pair<double, double> result = Montecarlo_integration(hyperrectangle, n, function, dim);
    }
    else
    {
        std::cout << "Invalid input.";
    }*/

    HyperSphere hypersphere(dim, rad);
    std::pair<double, double> result = Montecarlo_integration(hypersphere, n, function, dim);

    std::cout << std::endl
              << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;

    // calculate the exact domain if the user tries to integrate the volume of the hypersphere
    //TODO with various domains
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
