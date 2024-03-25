#ifndef ASSET_HPP
#define ASSET_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


class Asset {
public:
    double get_mean_return() const;
    double get_std_dev() const;
    double get_closing_price() const;
    std::string get_name() const;

    void set_mean_return(double mean_return);
    void set_std_dev(double std_dev);
    void set_closing_price(double closing_price);
    void set_name(std::string name);

private:
    std::string name;
    double mean_return;
    double std_dev;
    double closing_price;

};



#endif
