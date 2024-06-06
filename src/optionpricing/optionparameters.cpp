#include "../../include/optionpricing/optionparameters.hpp"

  // Function to create the payoff function for option pricing
  // in the finance-oriented project
std::pair<std::string, std::vector<double>> createPayoffFunction(double k, const std::vector<Asset> &assets)
{
      // Vector to store coefficients
    std::vector<double> coefficients;
    coefficients.reserve(assets.size());

      // Result pair containing function string and coefficients
    std::pair<std::string, std::vector<double>> result;
    std::string function = "max(0, (";

    for (size_t i = 0; i < assets.size(); ++i)
    {
          // Add coefficient to the vector
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

  // Function to calculate the strike price for option pricing
  // in the finance-oriented project.
  // Payoff calculated in a simple way by summing the last real values of the assets.
double calculateStrikePrice(const std::vector<Asset> &assets)
{
    double strike_price = 0.0;
    for (size_t i = 0; i < assets.size(); ++i)
    {
        strike_price += assets[i].getLastRealValue();
    }
    return strike_price;
}
