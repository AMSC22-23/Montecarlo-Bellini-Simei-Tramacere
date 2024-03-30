#ifndef PROJECT_OPTIONPRICING_
    #define PROJECT_OPTIONPRICING_

#include <vector>
#include "asset.hpp"

int predict_future_month(Asset& asset, std::vector<double>& prices, std::default_random_engine& generator);

int predict_price(Asset& asset, int iterations, std::default_random_engine& generator);

#endif