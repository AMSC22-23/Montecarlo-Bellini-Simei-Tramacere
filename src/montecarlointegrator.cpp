#include "../include/project/montecarlointegrator.hpp"
#include "../include/project/functionevaluator.hpp"

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperCube &hypercube)
{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    auto   start               = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(hypercube.getdimension());

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            hypercube.generate_random_point(random_point_vector);

            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();
            total_value         += result;
            total_squared_value += result * result;
        }
    }

      // calculate the integral
    hypercube.calculate_volume();
    double domain   = hypercube.get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

      // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;

      // stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperRectangle &hyperrectangle, 
    bool finance /* = false */, const std::vector<const Asset *> &assetPtrs /* = std::vector<const Asset*>() */, 
    double std_dev_from_mean /* = 5.0 */, double* variance /* = nullptr */, 
    std::vector<double> coefficients, int strike_price)
{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    auto   start               = std::chrono::high_resolution_clock::now();

    if (!finance)
    {
        std::vector<double> random_point_vector(hyperrectangle.getdimension());

#pragma omp parallel private(result)
        {
            mu::Parser parser;

#pragma omp for reduction(+ : total_value, total_squared_value)
            for (int i = 0; i < n; ++i)
            {
                hyperrectangle.generate_random_point(random_point_vector);
                if (!random_point_vector.empty())
                {
                    result = evaluateFunction(function, random_point_vector, parser);
                    parser.ClearVar();
                    total_value         += result;
                    total_squared_value += result * result;
                }
                else
                {
                    std::cout << "Error generating random point" << std::endl;
                    i--;
                }
            }
        }
    }
    else
    {

#pragma omp parallel
        {
            double total_value_thread = 0.0;
            double total_squared_value_thread = 0.0;
            std::vector<double> random_point_vector(assetPtrs.size());
#pragma omp for reduction(+ : total_value, total_squared_value)
            for (int i = 0; i < n; ++i)
            {

                hyperrectangle.generate_random_point(random_point_vector, finance, assetPtrs, std_dev_from_mean);
    
                if (!random_point_vector.empty())
                {

                    result = 0.0;
                    for (size_t i = 0; i < random_point_vector.size(); ++i) {
                        result += random_point_vector[i] * coefficients[i];
                    }

                    result = std::max(0.0, (result - strike_price));

                    total_value_thread         += result;
                    total_squared_value_thread += result * result;
                }
                else
                {
                    std::cout << "Error generating random point" << std::endl;
                    i--;
                }
            }

            #pragma omp critical
            {
                total_value += total_value_thread;
                total_squared_value += total_squared_value_thread;
            }
        }
    }

      // calculate the integral
    hyperrectangle.calculate_volume();
    double domain   = hyperrectangle.get_volume();
    double integral = total_value / static_cast<double>(n) * domain;

      // calculate the variance
    *variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    *variance = sqrt( *variance  / static_cast<double>(n) );

      // stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, static_cast<double>(duration.count()));
}

std::pair<double, double> Montecarlo_integration(int n, const std::string &function, HyperSphere &hypersphere)

{
    double total_value         = 0.0;
    double total_squared_value = 0.0;
    double result              = 0.0;
    auto   start               = std::chrono::high_resolution_clock::now();
    std::vector<double> random_point_vector(hypersphere.getdimension());

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value, total_squared_value)
        for (int i = 0; i < n; ++i)
        {
            hypersphere.generate_random_point(random_point_vector);

            if (random_point_vector[0] != 0.0)
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value         += result;
                total_squared_value += result * result;
            }
            else
            {
                i--;
            }
        }
    }

      // calculate the integral
    hypersphere.calculate_volume();
    double domain = hypersphere.get_volume();
    std::cout << "domain: " << domain << std::endl;
    double integral = total_value / static_cast<double>(n) * domain;

      // calculate the variance
    double variance = total_squared_value / static_cast<double>(n) - (total_value / static_cast<double>(n)) * (total_value / static_cast<double>(n));
    std::cout << "Variance: " << variance << std::endl;

      // stop the timer
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}
