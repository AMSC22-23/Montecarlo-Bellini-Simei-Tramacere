#ifndef ASSET_HPP
  #define ASSET_HPP

#include <string>


/**
 * @class Asset
 * @brief This class represents an asset, which is a financial instrument that can be traded.
 * 
 * In finance, an asset is any resource owned by an individual, corporation, or country 
 * that is expected to provide future economic benefits.
 * Assets are what an option is based on.
 */
class Asset
{
public: 
    /**
     * @brief Construct a new Asset object
     * @details Default constructor
     */
    Asset() = default;

    /**
     * @brief Construct a new Asset object
     * @details Custom constructor
     * @param name Name of the asset
     * @param return_mean Return mean of the asset
     * @param closing_price Last real value of the asset
     * @param return_std_dev Standard deviation of the return of the asset
     * @param expected_price Expected price of the asset
     * @param time_taken Time taken to compute the expected price
     */
    Asset(const std::string &name, double return_mean, double closing_price,
          double return_std_dev, double expected_price, double time_taken)
        :  name(name), return_mean(return_mean), closing_price(closing_price),
          return_std_dev(return_std_dev), expected_price(expected_price),
          time_taken(time_taken) {}

    /**
     * @brief Get the return mean of the asset 
     * @return A double representing the return mean of the asset
     * 
     * The return mean is the average of the returns of the asset.
     * A return is the profit or loss derived from investing in an asset.
     */
    double getReturnMean() const { return return_mean; }

    /**
     * @brief Get the name of the asset
     * @return A string representing the name of the asset
     */
    std::string getName() const { return name; }

    /**
     * @brief Get the standard deviation of the return of the asset
     * @return A double representing the standard deviation of the return of the asset
     * 
     * The standard deviation is a measure of the amount of variation or dispersion of a set of values.
     * In the case of an asset, it is a measure of the amount of variation of the return of the asset.
     */
    double getReturnStdDev() const { return return_std_dev; }

    /**
     * @brief Get the closing price of the asset
     * @return A double representing the closing price of the asset
     * 
     * The closing price is the last real value of the asset.
     * It is the price at which the asset was last traded.
     */
    double getLastRealValue() const { return closing_price; }

    /**
     * @brief Get the expected price of the asset
     * @return A double representing the expected price of the asset
     * 
     * The expected price is the price that the asset is expected to reach in the future.
     */
    double getExpectedPrice() const { return expected_price; }

    /**
     * @brief Set the return mean of the asset
     * @param return_mean A double representing the return mean of the asset
     */
    void setReturnMean(double return_mean) { this->return_mean = return_mean; }

    /**
     * @brief Set the name of the asset
     * @param name A string representing the name of the asset
     */
    void setName(const std::string &name) { this->name = name; }

    /**
     * @brief Set the standard deviation of the return of the asset
     * @param return_std_dev A double representing the standard deviation of the return of the asset
     */
    void setReturnStdDev(double return_std_dev) { this->return_std_dev = return_std_dev; }

    /**
     * @brief Set the closing price of the asset
     * @param closing_price A double representing the closing price of the asset
     */
    void setLastRealValue(double closing_price) { this->closing_price = closing_price; }

    /**
     * @brief Set the expected price of the asset
     * @param expected_price A double representing the expected price of the asset
     */
    void setExpectedPrice(double expected_price) { this->expected_price = expected_price; }

private: 
    std::string name;
    double return_mean    = 0.0;
    double closing_price  = 0.0;
    double return_std_dev = 0.0;
    double expected_price = 0.0;
    double time_taken     = 0.0;
};


#endif