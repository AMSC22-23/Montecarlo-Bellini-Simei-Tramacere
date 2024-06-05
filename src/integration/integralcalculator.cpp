#include "../../include/integration/integralcalculator.hpp"

  // Function to create the geometry object based on the domain type
Geometry *geometryFactory(size_t dim, double rad, double edge, std::vector<double> &hyper_rectangle_bounds, std::string domain_type)
{
    if (domain_type == "hr")
    {
        return new HyperRectangle(dim, hyper_rectangle_bounds);
    }
    else if (domain_type == "hs")
    {
        return new HyperSphere(dim, rad);
    }
    else if (domain_type == "hc")
    {
        return new HyperCube(dim, edge);
    }
    else
    {
        return nullptr;
    }
}

  // Function to compute the integral using the Monte Carlo method
void integralCalculator()
{
    size_t n, dim;
    double rad, edge, variance, standard_error = 0.0;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result(0.0, 0.0);
    bool success = false;

      // Get the input parameters
    buildIntegral(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

      // Create the geometry object based on the domain type
    std::unique_ptr<Geometry> geometry(geometryFactory(dim, rad, edge, hyper_rectangle_bounds, domain_type));

    if (geometry)
    {
        if (function == "1")
        {
              // if the function is 1, calculate the volume of the geometry
            geometry->calculateVolume();
            result.first = geometry->getVolume();
            if (result.first != 0.0)
                success = true;
        }
        else
        {
              // Calculate the integral using the Monte Carlo method
            result         = montecarloIntegration(n, function, *geometry, variance);
            standard_error = std::sqrt(variance / static_cast<double>(n)) * geometry->getVolume();
            if (result.first != 0.0 && result.second != 0.0)
                success = true;
        }
    }
    else
    {
        std::cout << "\nFail at creating class" << std::endl;
        return;
    }

      // Print the results
    if (success && function != "1")
    {
        std::cout << "\nThe approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;

          // Calculate the 95% confidence interval
        double lower_bound    = result.first - 1.96 * standard_error;
        double upper_bound    = result.first + 1.96 * standard_error;
        double interval_width = upper_bound - lower_bound;

        std::cout << "95% confidence interval: [" << lower_bound << ", " << upper_bound << "]" << std::endl;

          // Check if the confidence interval width is too large
        if (interval_width > 0.1 * result.first)
        {
            std::cout << "\nWarning: The result may be incorrect due to the confidence interval width\nbeing too large relative to the result." << std::endl;
            std::cout << "This may be due to the high variability of the integrated function." << std::endl;
        }
        std::cout << "\nThe time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
    }
    else if (success && function == "1")
    {
        std::cout << "\nThe exact result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    }
    else
    {
        std::cout << "\nMonte Carlo integration failed due to point generation errors\n"
                  << std::endl;
        std::cout << "result.first: " << result.first << std::endl;
    }
}
