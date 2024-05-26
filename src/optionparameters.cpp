#include "../include/project/optionparameters.hpp"

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

double calculateStrikePrice(const std::vector<Asset> &assets)
{
    // double strike_price = 0.0;
    // double spot_price   = 0.0;
    // double price_trend  = 0.0;
    // for (const auto &asset : assets)
    // {
    //     spot_price  += asset.getLastRealValue();
    //     price_trend += asset.getLastRealValue() * std::pow(1.0 + asset.getReturnMean(), 24);
    // }
    // double percentile_5_spot_price = spot_price * 0.05;
    // if (price_trend < spot_price - percentile_5_spot_price)
    // {
    //     strike_price = price_trend - 5.0;
    // }
    // else if (price_trend > spot_price + percentile_5_spot_price)
    // {
    //     strike_price = price_trend + 5.0;
    // }
    // else
    // {
    //     strike_price = spot_price;
    // }
    // return static_cast<int>(round(strike_price / 10.0)) * 10;
    double strike_price = 0.0;
    for( size_t i = 0; i < assets.size(); ++i )
    {
        strike_price += assets[i].getLastRealValue();
    }
    return strike_price;
}
