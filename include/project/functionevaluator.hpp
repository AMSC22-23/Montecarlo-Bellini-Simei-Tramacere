#ifndef PROJECT_FUNCTIONEVALUATOR_
    #define PROJECT_FUNCTIONEVALUATOR_

#include <string>
#include <vector>
#include "../muparser-2.3.4/include/muParser.h"
#include <chrono>

double evaluateFunction(const std::string &expression, const std::vector<double> &point, mu::Parser &parser);

#endif // PROJECT_FUNCTIONEVALUATOR_