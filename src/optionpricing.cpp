#include <random>
#include <chrono>
#include <omp.h>

#include "../include/project/asset.hpp"
#include "../include/project/optionpricing.hpp"


int predict_future_month(Asset& asset, std::vector<double>& prices, std::default_random_engine& generator) {
    try {
        double std_dev = asset.get_return_std_dev();
        double closing_price = asset.get_last_real_value();
        double mean = asset.get_return_mean();

        std::normal_distribution<double> distribution(mean, std_dev);

        prices.clear();
        prices.push_back(closing_price);
        double price;

        // Generate a new price for each day of the month
        for (int day = 0; day < 24; ++day) {
            price = prices[day] * (1 + distribution(generator));
            prices.push_back(price);
        }

        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int predict_price(Asset& asset, int iterations, std::default_random_engine& generator) {
    double sum = 0;
    int error_flag = 0;

    // #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < iterations; ++i) {
        std::vector<double> local_prices;
        int result = predict_future_month(asset, local_prices, generator);
        if (result == -1) {
            // #pragma omp atomic write
            error_flag = 1;
            continue;
        }
        sum += local_prices.back();
    }

    if (error_flag == 1) {
        return -1;
    }

    asset.set_expected_price(sum / iterations);
    return 0;
}