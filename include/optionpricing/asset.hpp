/**
 * @file asset.hpp
 * @brief This file contains the declaration of the Asset class.
 */

#ifndef ASSET_HPP
    #define ASSET_HPP

#include <string>
#include <vector>

/**
 * @class Asset
 * @brief Represents a financial asset that can be traded.
 *
 * In finance, an asset is any resource owned by an individual, corporation, or country
 * that is expected to provide future economic benefits.
 * Assets are the basis of options.
 */
class Asset
{
public:
    /**
     * @brief Default constructor for Asset.
     */
    Asset() = default;

    /**
     * @brief Custom constructor for Asset.
     * @param name Name of the asset.
     * @param return_mean Return mean of the asset.
     * @param closing_price Last real value of the asset.
     * @param return_std_dev Standard deviation of the return of the asset.
     * @param expected_price Expected price of the asset.
     */
    Asset(const std::string &name, double return_mean, double closing_price,
          double return_std_dev, double expected_price)
        : name(name), return_mean(return_mean), closing_price(closing_price),
          return_std_dev(return_std_dev), expected_price(expected_price) {}

    // Getters
    /**
     * @brief Get the return mean of the asset.
     * @return A double representing the return mean of the asset.
     */
    double getReturnMean() const { return return_mean; }

    /**
     * @brief Get the name of the asset.
     * @return A string representing the name of the asset.
     */
    std::string getName() const { return name; }

    /**
     * @brief Get the standard deviation of the return of the asset.
     * @return A double representing the standard deviation of the return of the asset.
     */
    double getReturnStdDev() const { return return_std_dev; }

    /**
     * @brief Get the closing price of the asset.
     * @return A double representing the closing price of the asset.
     */
    double getLastRealValue() const { return closing_price; }

    /**
     * @brief Get the expected price of the asset.
     * @return A double representing the expected price of the asset.
     */
    double getExpectedPrice() const { return expected_price; }

    /**
     * @brief Get the size of the vector containing daily returns of the asset.
     * @return A size_t representing the size of the vector.
     */
    size_t getDailyReturnsSize() const { return daily_returns.size(); }

    /**
     * @brief Get a specific daily return of the asset.
     * @param i Index of the daily return.
     * @return A double representing the daily return at index i.
     */
    double getDailyReturn(size_t i) const { return daily_returns[i]; }

    // Setters
    /**
     * @brief Set the return mean of the asset.
     * @param return_mean A double representing the return mean of the asset.
     */
    void setReturnMean(double return_mean) { this->return_mean = return_mean; }

    /**
     * @brief Set the name of the asset.
     * @param name A string representing the name of the asset.
     */
    void setName(const std::string &name) { this->name = name; }

    /**
     * @brief Set the standard deviation of the return of the asset.
     * @param return_std_dev A double representing the standard deviation of the return of the asset.
     */
    void setReturnStdDev(double return_std_dev) { this->return_std_dev = return_std_dev; }

    /**
     * @brief Set the closing price of the asset.
     * @param closing_price A double representing the closing price of the asset.
     */
    void setLastRealValue(double closing_price) { this->closing_price = closing_price; }

    /**
     * @brief Set the expected price of the asset.
     * @param expected_price A double representing the expected price of the asset.
     */
    void setExpectedPrice(double expected_price) { this->expected_price = expected_price; }

    /**
     * @brief Set the daily returns of the asset.
     * @param daily_returns A vector of doubles representing the daily returns of the asset.
     */
    void setDailyReturns(std::vector<double> daily_returns) { this->daily_returns = daily_returns; }

private:
    std::string name;
    double return_mean = 0.0;
    double closing_price = 0.0;
    double return_std_dev = 0.0;
    double expected_price = 0.0;
    std::vector<double> daily_returns = {};
};

#endif
