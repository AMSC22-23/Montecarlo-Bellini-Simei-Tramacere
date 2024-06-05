#include "../include/inputmanager.hpp"

void buildIntegral(size_t &n, size_t &dim, double &rad, double &edge, std::string &function,
                   std::string &domain_type, std::vector<double> &hyper_rectangle_bounds)
{
      // Read and validate domain type
    readValidatedInput<std::string>("Insert the type of domain you want to integrate:\n  hc - hyper-cube\n  hs - hyper-sphere\n  hr - hyper-rectangle\n",
                                    domain_type,
                                    [](const std::string &val)
                                    { return val == "hs" || val == "hr" || val == "hc"; });

      // Read and validate number of random points
    readValidatedInput<size_t>("Insert the number of random points to generate:\n", n, [](const size_t &val)
                               { return val > 0; });

    if (domain_type == "hs")
    {
          // Read and validate hypersphere dimension
        readValidatedInput<size_t>("Insert the dimension of the hypersphere:\n", dim, [](const size_t &val)
                                   { return val > 0; });

          // Read and validate hypersphere radius
        readValidatedInput<double>("Insert the radius of the hypersphere:\n", rad, [](const double &val)
                                   { return val > 0; });
    }
    else if (domain_type == "hr")
    {
          // Read and validate hyperrectangle dimension
        readValidatedInput<size_t>("Insert the dimension of the hyperrectangle:\n", dim, [](const size_t &val)
                                   { return val > 0; });

          // Reserve space for bounds
        hyper_rectangle_bounds.reserve(dim * 2);

          // Read and validate hyperrectangle bounds
        for (size_t i = 0; i < 2 * dim; ++i)
        {
            double tmp;
            size_t      current_dimension = i / 2 + 1;
            std::string boundType         = (i % 2 == 0) ? "lower" : "upper";
            std::string suffix;

              // Determine the ordinal suffix
            if (current_dimension % 10 == 1 && current_dimension % 100 != 11)
                suffix = "st";
            else if (current_dimension % 10 == 2 && current_dimension % 100 != 12)
                suffix = "nd";
            else if (current_dimension % 10 == 3 && current_dimension % 100 != 13)
                suffix = "rd";
            else
                suffix = "th";

              // Read and validate each bound
            readValidatedInput<double>("Insert the " + boundType + " bound of the " + std::to_string(current_dimension) + suffix + " dimension of the hyper-rectangle:\n",
                                       tmp,
                                       [&](const double &val)
                                       { return boundType == "lower" || val > hyper_rectangle_bounds[i - 1]; });

            hyper_rectangle_bounds.emplace_back(tmp);
        }
    }
    else if (domain_type == "hc")
    {
          // Read and validate hypercube dimension
        readValidatedInput<size_t>("Insert the dimension of the hypercube:\n", dim, [](const size_t &val)
                                   { return val > 0; });

          // Read and validate hypercube edge length
        readValidatedInput<double>("Insert the edge length of the hypercube:\n", edge, [](const double &val)
                                   { return val > 0; });
    }

      // Read function to integrate
    std::cout << "Insert the function to integrate:\n";
    readInput(std::cin, function);
}
