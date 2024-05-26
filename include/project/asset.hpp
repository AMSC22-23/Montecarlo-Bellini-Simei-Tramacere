#ifndef ASSET_HPP
#define ASSET_HPP

#include <string>

class Asset
{
public: 
      // Constructors
    Asset() = default;
    Asset(const std::string &name, double return_mean, double closing_price,
          double return_std_dev, double expected_price, double time_taken)
        :  name(name), return_mean(return_mean), closing_price(closing_price),
          return_std_dev(return_std_dev), expected_price(expected_price),
          time_taken(time_taken) {}

      // Getters
    double getReturnMean() const { return return_mean; }
    std::string getName() const { return name; }
    double getReturnStdDev() const { return return_std_dev; }
    double getLastRealValue() const { return closing_price; }
    double getExpectedPrice() const { return expected_price; }
    double getTimeTaken() const { return time_taken; }

      // Setters
    void setReturnMean(double return_mean) { this->return_mean = return_mean; }
    void setName(const std::string &name) { this->name = name; }
    void setReturnStdDev(double return_std_dev) { this->return_std_dev = return_std_dev; }
    void setLastRealValue(double closing_price) { this->closing_price = closing_price; }
    void setExpectedPrice(double expected_price) { this->expected_price = expected_price; }
    void setTimeTaken(double time_taken) { this->time_taken = time_taken; }

private: 
    std::string name;
    double return_mean    = 0.0;
    double closing_price  = 0.0;
    double return_std_dev = 0.0;
    double expected_price = 0.0;
    double time_taken     = 0.0;
};

#endif
