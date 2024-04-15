#include "../include/project/inputmanager.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

void input_manager(int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds)
{
    // Get and validate number of random points
    std::cout << "Insert the type of domain you want to integrate: (hc for hyper-cube, hs for hyper-sphere, hr for hyper-rectangle)" << std::endl;
    std::cin >> domain_type;
    while (std::cin.fail() || (domain_type != "hs" && domain_type != "hr" && domain_type != "hc"))
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter hs for hyper-sphere, hr for hyper-rectangle: ";
        std::cin >> domain_type;
    }

    std::cout << "Insert the number of random points to generate: ";
    std::cin >> n;
    while (std::cin.fail() || n <= 0)
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter a positive integer: ";
        std::cin >> n;
    }

    if (domain_type == "hs")
    {
        std::cout << "Insert the dimension of the hypersphere: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }

        std::cout << "Insert the radius of the hypersphere: ";
        std::cin >> rad;
        while (std::cin.fail() || rad <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> rad;
        }
    }
    else if (domain_type == "hr")
    {
        std::cout << "Insert the dimension of the hyperrectangle: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }

        hyper_rectangle_bounds.reserve(dim*2);
        double tmp;

        for (int i = 0; i < 2 * dim; i++)
        {
            if (i == 0)
                std::cout << "Insert the 1st dimension coordinate of the hyper-rectangle: ";
            else if (i == 1)
                std::cout << "Insert the 2nd dimension coordinate of the hyper-rectangle: ";
            else if (i == 2)
                std::cout << "Insert the 3rd dimension coordinate of the hyper-rectangle: ";
            else
                std::cout << "Insert the " << i+1 << "th dimension coordinate of the hyper-rectangle: ";

            std::cin >> tmp;
            hyper_rectangle_bounds.push_back(tmp);
            while (std::cin.fail())
            {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a positive number: ";
                std::cin >> hyper_rectangle_bounds[i];
            }
        }
    }
    else if (domain_type == "hc")
    {
        std::cout << "Insert the dimension of the hypercube: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }
        std::cout << "Insert the edge length of the hypercube: ";
        std::cin >> edge;
        while (std::cin.fail() || edge <= 0)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> edge;
        }
    }

    // ask the user to insert the function to integrate
    std::cout << "Insert the function to integrate: ";
    std::cin >> function;
}

std::string create_function(int k, const std::vector<Asset> &assets)
{
    std::string function = "max(0, (";

    for (size_t i = 0; i < assets.size(); ++i)
    {
        function += "x" + std::to_string(i + 1) + " * " + std::to_string(assets[i].get_last_real_value());
        if (i < assets.size() - 1)
        {
            function += " + ";
        }
    }

    function += ") - " + std::to_string(k) + ")";
    return function;
}

int calculate_strike_price(const std::vector<Asset> &assets)
{
    double strike_price = 0.0;
    double spot_price = 0.0;
    double price_trend = 0.0;
    for (const auto &asset : assets)
    {
        spot_price += asset.get_last_real_value();
        price_trend += asset.get_last_real_value() * std::pow(1.0 + asset.get_return_mean(), 24);

        // debug
        std::cout << "The spot price of the asset is: " << spot_price << std::endl;
        std::cout << "The raw expected price of the asset is: " << price_trend << std::endl;
    }
    double percentile_5_spot_price = spot_price * 0.05;
    if (price_trend < spot_price - percentile_5_spot_price)
    {
        strike_price = price_trend - 5.0;
    }
    else if (price_trend > spot_price + percentile_5_spot_price)
    {
        strike_price = price_trend + 5.0;
    }
    else
    {
        strike_price = spot_price;
    }
    return static_cast<int>(round(strike_price/ 10.0)) * 10;
}

int set_integration_bounds(std::vector<double> &integration_bounds, const std::vector<Asset> &assets, int std_dev_from_mean /* = 24 */)
{
    try
    {
        int j = 0;
        for (size_t i = 0; i < assets.size() * 2 - 1; i += 2)
        {
            integration_bounds[i]     = assets[j].get_return_mean() - std_dev_from_mean * assets[j].get_return_std_dev() + 1.0;
            integration_bounds[i + 1] = assets[j].get_return_mean() + std_dev_from_mean * assets[j].get_return_std_dev() + 1.0;
            j++;
        }
    }
    catch (const std::exception &e)
    {
        return -1;
    }
    return 0;
}