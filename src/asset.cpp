#include "../include/project/asset.hpp"

// Getters
double Asset::get_return_mean() const {
    return return_mean;
}

std::string Asset::get_name() const {
    return name;
}

double Asset::get_return_std_dev() const {
    return return_std_dev;
}

double Asset::get_last_real_value() const {
    return closing_price;
}

double Asset::get_expected_price() const {
    return expected_price;
}

double Asset::get_time_taken() const {
    return time_taken;
}

// Setters
void Asset::set_return_mean(double return_mean) {
    this->return_mean = return_mean;
}

void Asset::set_name(std::string name) {
    this->name = name;
}

void Asset::set_return_std_dev(double return_std_dev) {
    this->return_std_dev = return_std_dev;
}

void Asset::set_last_real_value(double last_real_value) {
    this->closing_price = last_real_value;
}

void Asset::set_expected_price(double expected_price) {
    this->expected_price = expected_price;
}

void Asset::set_time_taken(double time_taken) {
    this->time_taken = time_taken;
}