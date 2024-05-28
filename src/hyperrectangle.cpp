#include "../include/project/hyperrectangle.hpp"


// Constructor
HyperRectangle::HyperRectangle(size_t dim, std::vector<double> &hyper_rectangle_bounds)
    : hyper_rectangle_bounds(hyper_rectangle_bounds), volume(1.0), dimension(dim), eng(rd()) {}

// Function to generate a random point in the hyperrectangle domain
// for the Monte Carlo method of the original project
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

// Function to generate a random point in the hyperrectangle domain
// for the finance oriented project
void HyperRectangle::financeGenerateRandomPoint(std::vector<double> &random_point,
                                                const std::vector<const Asset *> &assetPtrs,
                                                const double std_dev_from_mean)
{
    try
    {
        thread_local std::mt19937 eng(std::random_device{}());

#pragma omp parallel for 
        for (size_t i = 0; i < assetPtrs.size(); ++i)
        {
            std::normal_distribution<double> distribution(assetPtrs[i]->getReturnMean(), assetPtrs[i]->getReturnStdDev());
            double price = assetPtrs[i]->getLastRealValue();
                price = price * (1 + distribution(eng));
            double predicted_return = price / assetPtrs[i]->getLastRealValue();
#pragma omp critical
                {
                    random_point[i] = predicted_return;
                }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        random_point.clear();
    }
}