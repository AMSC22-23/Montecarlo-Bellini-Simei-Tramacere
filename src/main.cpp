#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>

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
#include "project/asset.hpp"


int main()
{
    /*int n, dim;
    double rad, edge;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result = {0.0, 0.0};*/
    

    Asset asset{};
    
    std::string filename = "NVDA.csv";
    int csv_result = csv_reader(filename, &asset);
    if (csv_result == -1) {
        std::cout << "Error reading the file." << std::endl;
        return -1;
    }

    // printing the stored data
    
    std::cout << "Mean Return: " << asset.get_mean_return() << std::endl;

    // get the input from the user
    /*input_manager(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

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
        std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;*/

    
    return 0;
}
