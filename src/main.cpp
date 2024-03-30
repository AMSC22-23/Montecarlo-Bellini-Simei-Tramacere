#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "../include/muparser-2.3.4/include/muParser.h"
#include "../include/muparser-2.3.4/include/muParserIncluder.h"

#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/montecarlointegrator.hpp"
#include "../include/project/csvhandler.hpp"
#include "../include/project/optionpricing.hpp"

int main()
{
    // Load the assets from the csv file
    std::vector<Asset> assets;
    int csv_result = load_assets_from_csv("../csv/", assets);
    if (csv_result == -1)
    {
        std::cout << "Error loading the assets from the CSV files" << std::endl;
        return -1;
    }

    // Print the assets
    for (const auto &asset : assets)
    {
        std::cout << "Asset: " << asset.get_name() << std::endl;
        std::cout << "Return Mean: " << asset.get_return_mean() << std::endl;
        std::cout << "Return standard deviation: " << asset.get_return_std_dev() << std::endl;
        std::cout << std::endl;
    }

    // Predict the future month for each asset
    int iterations = 1e7;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (auto &asset : assets)
    {
        std::random_device rd;
        std::default_random_engine generator(rd());
        auto start = std::chrono::high_resolution_clock::now();
        int result = predict_price(asset, iterations, generator);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        asset.set_time_taken(duration.count() * 1e-6);

        if (result == -1)
        {
            #pragma omp critical
            std::cout << "Error predicting the future month for asset " << asset.get_name() << std::endl;
            continue;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Print the assets
    double variation_percentage = 0.0;

    for (const auto &asset : assets)
    {
        std::cout << "Expected price for asset " << asset.get_name() << ":" << std::endl;
        std::cout << asset.get_expected_price();
        variation_percentage = (asset.get_expected_price() - asset.get_last_real_value()) / asset.get_last_real_value() * 100;
        if (variation_percentage >= 0)
            std::cout << " (+" << std::setprecision(4) << variation_percentage << "%)" << std::endl;
        std::cout << "Time taken: " << std::setprecision(6) << asset.get_time_taken() << " seconds" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Total time taken: " << duration.count() * 1e-6 << " seconds" << std::endl;

    /*std::cout << "===============================================================================================================" << std::endl;
    std::cout << std::endl;

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
            result = Montecarlo_integration(n, function, hypersphere);
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
            result = Montecarlo_integration(n, function, hypercube);
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
            // result = Montecarlo_integration_hyperrectangle(n, function, dim, hyperrectangle);
            result = Montecarlo_integration(n, function, hyperrectangle);
        }
    }

    std::cout << std::endl
              << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    if (result.second != 0.0)
        std::cout << "The time needed to calculate the integral is: " << result.second * 1e-6 << " seconds" << std::endl;
    */
    return 0;
}
