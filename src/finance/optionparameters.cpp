#include "../../include/project/optionparameters.hpp"


// Function to create the payoff function for the option pricing
// for the finance oriented project
std::pair<std::string, std::vector<double>> createPayoffFunction(double k, const std::vector<Asset> &assets)
{
    std::vector<double> coefficients;
    coefficients.reserve(assets.size());

    std::pair<std::string, std::vector<double>> result;
    std::string function = "max(0, (";

    for (size_t i = 0; i < assets.size(); ++i)
    {
        coefficients.emplace_back(assets[i].getLastRealValue());
        function += "x" + std::to_string(i + 1) + " * " + std::to_string(coefficients[i]);
        if (i < assets.size() - 1)
        {
            function += " + ";
        }
    }

    function += ") - " + std::to_string(k) + ")";

    return std::make_pair(function, coefficients);
}

// Function to calculate the strike price for the option pricing
// for the finance oriented project
double calculateStrikePrice(const std::vector<Asset> &assets)
{
    double strike_price = 0.0;
    for( size_t i = 0; i < assets.size(); ++i )
    {
        strike_price += assets[i].getLastRealValue();
    }
    return strike_price;
}