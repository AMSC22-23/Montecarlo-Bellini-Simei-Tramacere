#ifndef ASSET_HPP
#define ASSET_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


class Asset {
public:
    double get_return_mean() const;
    std::string get_name() const;
    double get_return_std_dev() const;
    double get_last_real_value() const;
    double get_expected_price() const;
    double get_time_taken() const;

    void set_return_mean(double return_mean);
    void set_name(std::string name);
    void set_return_std_dev(double return_std_dev);
    void set_last_real_value(double last_real_value);
    void set_expected_price(double expected_price);
    void set_time_taken(double time_taken);

private:
    std::string name;
    double return_mean;
    double closing_price;
    double return_std_dev;
    double expected_price;
    double time_taken;
};

#endif
