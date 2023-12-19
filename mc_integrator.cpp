#include "mc_integrator.hpp"

#include <chrono>




double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser)
{
#pragma omp parallel for
    for (size_t i = 0; i < point.size(); ++i)
    {
        std::string varName = "x" + std::to_string(i + 1);
        parser.DefineVar(varName, const_cast<mu::value_type *>(&point[i]));
    }

    parser.SetExpr(expression);

    try
    {
        double result = parser.Eval();
        // std::cout << "Parsed Function: " << parser.GetExpr() << std::endl;
        // std::cout << "Intermediate result: " << result << std::endl;
        return result;
    }
    catch (mu::Parser::exception_type &e)
    {
        std::cout << "Error evaluating expression: " << e.GetMsg() << std::endl;
        return 0.0; // Return some default value in case of an error
    }
}

std::pair<double, double> Montecarlo_integration(HyperSphere &integration_domain, int n, const std::string &function, int dimension)
{
    double total_value = 0.0;
    double result = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> random_point_vector(dimension);

#pragma omp parallel private(result)
    {
        mu::Parser parser;
#pragma omp for reduction(+ : total_value)
        for (int i = 0; i < n; ++i)
        {
            integration_domain.generate_random_point(random_point_vector);
            if (!random_point_vector.empty())
            {
                result = evaluateFunction(function, random_point_vector, parser);
                parser.ClearVar();
                total_value += result;
                integration_domain.add_point_inside();
            }
        }
    }

    // calculate the integral
    integration_domain.calculate_approximated_volume(n);
    double domain = integration_domain.get_approximated_volume();
    int points_inside = integration_domain.get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}