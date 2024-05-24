#include "../include/project/hyperrectangle.hpp"

HyperRectangle::HyperRectangle(size_t dim, std::vector<double> &hyper_rectangle_bounds)
    : hyper_rectangle_bounds(hyper_rectangle_bounds), volume(1.0), dimension(dim), eng(rd()) {}

void HyperRectangle::generateRandomPoint(std::vector<double> &random_point)
{
    int j = 0;
    for (size_t i = 0; i < dimension * 2 - 1; i += 2)
    {
        std::uniform_real_distribution<double> distribution(hyper_rectangle_bounds[i], hyper_rectangle_bounds[i + 1]);
        random_point[j] = distribution(eng);
        j++;
    }
}

void HyperRectangle::financeGenerateRandomPoint(std::vector<double> &random_point, const std::vector<const Asset *> &assetPtrs, const double std_dev_from_mean)
{
    try
    {
        thread_local std::mt19937 eng(std::random_device{}());

#pragma omp parallel for
        for (size_t i = 0; i < assetPtrs.size(); ++i)
        {
            std::normal_distribution<double> distribution(assetPtrs[i]->getReturnMean(), assetPtrs[i]->getReturnStdDev());
            double price = assetPtrs[i]->getLastRealValue();

              // Generate a new return for each day of the month
            // for (size_t day = 0; day < 24; ++day)
            // {
                price = price * (1 + distribution(eng));
            // }
            
              // Check if the return is within the bounds
            double predicted_return = price / assetPtrs[i]->getLastRealValue();
            if (predicted_return > assetPtrs[i]->getReturnMean() + std_dev_from_mean * assetPtrs[i]->getReturnStdDev() + 1.0 ||
                predicted_return < assetPtrs[i]->getReturnMean() - std_dev_from_mean * assetPtrs[i]->getReturnStdDev() + 1.0)
            {
                i--;
                continue;
            }
            else
            {
#pragma omp critical
                {
                    random_point[i] = predicted_return;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        random_point.clear();
    }
}

