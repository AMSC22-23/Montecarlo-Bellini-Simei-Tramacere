#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>

#include "muParser.h"
#include "muParser.cpp"
#include "muParserBase.cpp"
#include "muParserBytecode.cpp"
#include "muParserCallback.cpp"
#include "muParserError.cpp"
#include "muParserTokenReader.cpp"
#include "HyperSphere.cpp"
// note that the following method implements rn only the special case with the function f(x) = 1
// TODO: generalize the method for all the functions

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

std::pair<double, double> Montecarlo_integration(HyperSphere &hypersphere, int n, const std::string &function, int dimension)
{

    static mu::Parser parser; // Declare the parser as static to reuse it for each point
    double total_value = 0.0;
    double result = 0.0;
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> random_point_vector;
    random_point_vector.reserve(dimension);

// generate the random points
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        random_point_vector.clear();
        hypersphere.generate_random_point(random_point_vector);
        if (!random_point_vector.empty())
        {
            result = evaluateFunction(function, random_point_vector, parser);
            parser.ClearVar();
            total_value += result;
            hypersphere.add_point_inside();
        }
    }

    // calculate the integral
    hypersphere.calculate_approximated_volume(n);
    double domain = hypersphere.get_approximated_volume();
    int points_inside = hypersphere.get_points_inside();
    double integral = total_value / static_cast<double>(points_inside) * domain;
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}

int main(int argc, char **argv)
{
    int n, dim;
    double rad;
    std::string function;
    // ask the user to insert the parameters
    std::cout << "Insert the number of random points to generate: ";
    std::cin >> n;
    std::cout << "Insert the dimension of the hypersphere: ";
    std::cin >> dim;
    std::cout << "Insert the radius of the hypersphere: ";
    std::cin >> rad;
    // ask the user to insert the function to integrate
    std::cout << "Insert the function to integrate: ";
    std::cin >> function;
    // generate the hypersphere
    HyperSphere hypersphere(dim, rad);
    // calculate the integral and print the results
    std::pair<double, double> result = Montecarlo_integration(hypersphere, n, function, dim);
    std::cout << std::endl
              << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;
    // hypersphere.calculate_volume();
    // double exact_domain = hypersphere.get_volume();
    // std::cout << "The exact domain in " << dim << " dimensions of your integral is: " << exact_domain << std::endl;
    // std::cout << "The absolute error is: " << std::abs(result.first - volume) << std::endl;
    // std::cout << "The relative error is: " << std::abs(result.first - volume) / volume << std::endl;
    return 0;
}
