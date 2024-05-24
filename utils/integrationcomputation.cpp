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
    size_t n, dim;
    double rad, edge, variance, standard_error = 0.0;
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
            result         = montecarloIntegration(n, function, hypersphere, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * hypersphere.getVolume();
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
            result         = montecarloIntegration(n, function, hypercube, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * hypercube.getVolume();
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
            result         = montecarloIntegration(n, function, hyperrectangle, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * hyperrectangle.getVolume();
        }
    }

    std::cout << "\nThe approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;

    double lower_bound    = result.first - 1.96 * standard_error;
    double upper_bound    = result.first + 1.96 * standard_error;
    double interval_width = upper_bound - lower_bound;

    std::cout << "95% confidence interval: [" << lower_bound << ", " << upper_bound << "]" << std::endl;

    if (interval_width > 0.1 * result.first)
    {
        std::cout << "\nWarning: The confidence interval width is large relative to the result." << std::endl;
        std::cout << "This may be due to the high variability of the integrated function." << std::endl;
    }
    if (result.second != 0.0)
        std::cout << "\nThe time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
    else
        std::cout << "\nERROR: The time needed to calculate the integral is 0.0 seconds" << std::endl;
}
