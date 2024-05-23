#include <iostream>
#include <vector>
#include <string>

#include "../include/project/integrationcomputation.hpp"
#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/montecarlo.hpp"

void integrationComputation()
{
    int n, dim;
    double rad, edge, variance;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result = {0.0, 0.0};

    buildIntegral(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

    if (domain_type == "hs")
    {
        HyperSphere hypersphere(dim, rad);
        if (function == "1")
        {
            hypersphere.calculateVolume();
            result.first = hypersphere.getVolume();
        }
        else
        {
            result = montecarloIntegration(n, function, hypersphere, variance);
        }
    }
    else if (domain_type == "hc")
    {
        HyperCube hypercube(dim, edge);
        if (function == "1")
        {
            hypercube.calculateVolume();
            result.first = hypercube.getVolume();
        }
        else
        {
            result = montecarloIntegration(n, function, hypercube, variance);
        }
    }
    else if (domain_type == "hr")
    {
        HyperRectangle hyperrectangle(dim, hyper_rectangle_bounds);
        if (function == "1")
        {
            hyperrectangle.calculateVolume();
            result.first = hyperrectangle.getVolume();
        }
        else
        {
            result = montecarloIntegration(n, function, hyperrectangle, variance);
        }
    }

    std::cout << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    if (result.second != 0.0)
        std::cout << "The time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
    else
        std::cout << "ERROR: The time needed to calculate the integral is 0.0 seconds" << std::endl;
    std::cout << "The variance of the integral is: " << variance << std::endl;
}
