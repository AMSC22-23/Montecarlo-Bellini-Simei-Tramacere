#include <iostream>
#include <vector>
#include <string>

#include "../include/project/integrationcomputation.hpp"
#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/montecarlo.hpp"


// Function that evaluates the integral of a function in a hypersphere, hypercube or hyperrectangle
// domain using the Monte Carlo method
// This function is the core od the original project
void integrationComputation()
{
    size_t n, dim;
    double rad, edge, variance, standard_error = 0.0;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result = {0.0, 0.0};

    // Get the input values from the user
    buildIntegral(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

    // Calculate the integral over the chosen domain
    // Check if the domain is a hypersphere
    if (domain_type == "hs")
    {
        HyperSphere hypersphere(dim, rad);
        if (function == "1")
        {
            // Calculate the volume of the hypersphere by the formula V = pi^(d/2) * r^d / (d/2)!
            // if the function is 1
            hypersphere.calculateVolume();
            result.first = hypersphere.getVolume();
        }
        else
        {
            // Calculate the integral using the Monte Carlo method
            // if the function is not 1
            result         = montecarloIntegration(n, function, hypersphere, variance);
            standard_error = sqrt(variance / static_cast<double>(n)) * hypersphere.getVolume();
        }
    }

    // Check if the domain is a hypercube
    else if (domain_type == "hc")
    {
        HyperCube hypercube(dim, edge);
        if (function == "1")
        {
            // Calculate the volume of the hypercube by the formula V = edge^d if the function is 1
            hypercube.calculateVolume();
            result.first = hypercube.getVolume();
        }
        else
        {
            // Calculate the integral using the Monte Carlo method if the function is not 1
            result         = montecarloIntegration(n, function, hypercube, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * hypercube.getVolume();
        }
    }

    // Check if the domain is a hyperrectangle
    else if (domain_type == "hr")
    {
        HyperRectangle hyperrectangle(dim, hyper_rectangle_bounds);
        if (function == "1")
        {
            // Calculate the volume of the hyperrectangle by the formula V = edge^d if the function is 1
            hyperrectangle.calculateVolume();
            result.first = hyperrectangle.getVolume();
        }
        else
        {
            // Calculate the integral using the Monte Carlo method if the function is not 1
            result         = montecarloIntegration(n, function, hyperrectangle, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * hyperrectangle.getVolume();
        }
    }

    // Print the approximate result of the integral calculated using the Monte Carlo method
    std::cout << "\nThe approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;

    double lower_bound    = result.first - 1.96 * standard_error;
    double upper_bound    = result.first + 1.96 * standard_error;
    double interval_width = upper_bound - lower_bound;

    // Print the 95% confidence interval
    std::cout << "95% confidence interval: [" << lower_bound << ", " << upper_bound << "]" << std::endl;

    if (interval_width > 0.1 * result.first)
    {
        std::cout << "\nWarning: The result may be incorrect due to the confidence interval width\nbeing too large relative to the result." << std::endl;
        std::cout << "This may be due to the high variability of the integrated function." << std::endl;
    }
    if (result.second != 0.0)
        // Print the time needed to calculate the integral
        std::cout << "\nThe time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
    else
        std::cout << "\nERROR: The time needed to calculate the integral is 0.0 seconds" << std::endl;
}
