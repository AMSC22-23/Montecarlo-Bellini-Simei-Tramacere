#include <omp.h>
#include <iostream>
#include "../include/project/hyperrectangle.hpp"

constexpr double PI = 3.14159265358979323846;

HyperRectangle::HyperRectangle(int dim, std::vector<double> &hyper_rectangle_bounds): eng(rd())
{
    dimension                    = dim;
    this->hyper_rectangle_bounds = hyper_rectangle_bounds;
    volume                       = 1.0;
}

void HyperRectangle::generate_random_point(std::vector<double> &random_point, bool finance, const std::vector<const Asset*>& assetPtrs, double std_dev_from_mean /* = 5.0 */)
{
    if (!finance)
    {
        int j = 0;

// #pragma omp parallel for
    for (int i = 0; i < dimension * 2 - 1; i += 2)
    {
        std::uniform_real_distribution<double> distribution(hyper_rectangle_bounds[i], hyper_rectangle_bounds[i + 1]);
        random_point[j] = distribution(eng);
        j++;
    }
    }
    else
    {
        try
        {
            thread_local std::mt19937 eng(std::random_device{}());

            #pragma omp parallel for
            for (size_t i = 0; i < assetPtrs.size(); i++)
            {
                std::normal_distribution<double> distribution(assetPtrs[i]->get_return_mean(), assetPtrs[i]->get_return_std_dev());
                double price = assetPtrs[i]->get_last_real_value();

                // Generate a new return for each day of the month
                for (int day = 0; day < 24; ++day)
                {
                    price = price * (1 + distribution(eng));
                }

                // Check if the return is within the bounds
                double normalized_price = price / assetPtrs[i]->get_last_real_value();
                if (normalized_price > assetPtrs[i]->get_return_mean() + std_dev_from_mean * assetPtrs[i]->get_return_std_dev() + 1.0 ||
                    normalized_price < assetPtrs[i]->get_return_mean() - std_dev_from_mean * assetPtrs[i]->get_return_std_dev() + 1.0)
                {
                    i--;
                    // std::cout << "Price out of bounds" << std::endl;
                    continue;  
                }
                else
                {
                    #pragma omp critical
                    {
                        random_point[i] = normalized_price;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error occurred: " << e.what() << std::endl;
            random_point.clear();
        }
}
}

void HyperRectangle::calculate_volume()
{
    for (int i = 0; i < 2 * dimension - 1; i += 2)
    {
        volume *= (hyper_rectangle_bounds[i + 1] - hyper_rectangle_bounds[i]);
    }
}


int HyperRectangle::getdimension() { return dimension; }
