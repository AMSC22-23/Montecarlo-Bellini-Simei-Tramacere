#ifndef PROJECT_FUNCTIONEVALUATOR_
    #define PROJECT_FUNCTIONEVALUATOR_

#include <string>
#include <vector>
#include <chrono>

#include "../../external/muparser-2.3.4/include/muParser.h"

  /**
 * @brief This function is used to evaluate a function
 * @details The function evaluates a function using the muParser library
 * @param expression The expression of the function
 * @param point The point to use in the function
 * @param parser The muParser object
 * @return The value of the function
 */
double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser);

#endif