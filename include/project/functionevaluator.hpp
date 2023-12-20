#ifndef PROJECT_FUNCTIONEVALUATOR_
#define PROJECT_FUNCTION_EVALUATOR_

#include <string>
#include <vector>
#include "muParser.h"
#include <chrono>

double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser);

#endif // PROJECT_FUNCTIONEVALUATOR_