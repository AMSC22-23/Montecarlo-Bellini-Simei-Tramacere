#include "../include/project/functionevaluator.hpp"
#include <chrono>

  // Function to evaluate the function using the muParser library
double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser)
{
      // Define the variables in the expression
    for (size_t i = 0; i < point.size(); ++i)
    {
        std::string varName = "x" + std::to_string(i + 1);
        parser.DefineVar(varName, const_cast<mu::value_type *>(&point[i]));
    }

      // Set the expression
    parser.SetExpr(expression);

      // Evaluate the expression
    try
    {
        double result = parser.Eval();
        return result;
    }
    catch (mu::Parser::exception_type &e)
    {
        std::cout << "Error evaluating expression: " << e.GetMsg() << std::endl;
          // Return some default value in case of an error
        return 0.0;
    }
}