#include "project/asset.hpp"

// Getters
double Asset::get_mean_return() const {
    return mean_return;
}

double Asset::get_std_dev() const {
    return std_dev;
}

double Asset::get_closing_price() const {
    return closing_price;
}

std::string Asset::get_name() const {
    return name;
}

// Setters
void Asset::set_mean_return(double mean_return) {
    this->mean_return = mean_return;
}

void Asset::set_std_dev(double std_dev) {
    this->std_dev = std_dev;
}

void Asset::set_closing_price(double closing_price) {
    this->closing_price = closing_price;
}

void Asset::set_name(std::string name) {
    this->name = name;
}
