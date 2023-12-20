#include "project/functionevaluator.hpp"
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
        return result;
    }
    catch (mu::Parser::exception_type &e)
    {
        std::cout << "Error evaluating expression: " << e.GetMsg() << std::endl;
        return 0.0; // Return some default value in case of an error
    }
}