#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>
//#include <cblas.h>

#include "../include/muparser-2.3.4/include/muParser.h"
#include "../include/muparser-2.3.4/include/muParserIncluder.h"

#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/montecarlo.hpp"
#include "../include/project/asset.hpp"



int main() {

    int n, dim;
    double rad, edge;
    std::string function;
    std::string domain_type;
    std::vector<double> hyper_rectangle_bounds;
    std::pair<double, double> result = {0.0, 0.0};
    
    /*Asset asset{};
    
    std::string filename = "NVDA.csv";
    int csv_result = csv_reader(filename, &asset);
    if (csv_result == -1) {
        std::cout << "Error reading the file." << std::endl;
        return -1;
    }*/

    // Get the input from the user
    input_manager(n, dim, rad, edge, function, domain_type, hyper_rectangle_bounds);

    // Apply the Monte Carlo method to the HyperSphere
    if (domain_type == "hs") {
        HyperSphere hs(dim, rad);
        result = hs_montecarlo_integration(hs, n, function, dim);
    }

    // Apply the Monte Carlo method to the HyperCube
    else if (domain_type == "hc") {
        HyperCube hc(dim, edge);
        result = hc_montecarlo_integration(hc, n, function, dim);
    }

    // Apply the Monte Carlo method to the HyperRectangle
    else if (domain_type == "hr") {
        HyperRectangle hr(dim, hyper_rectangle_bounds);
        result = hr_montecarlo_integration(hr, n, function, dim);
    }

    // If the user entered an invalid domain type
    else {
        std::cout << "Invalid input.";
        return -1;
    }

    // Print the result
    std::cout << std::endl
              << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    if (result.second != 0.0) 
        std::cout << "The time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;

    return 0;
}
