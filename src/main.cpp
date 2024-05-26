#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <fstream>

#include "../include/muparser-2.3.4/include/muParser.h"
#include "../include/muparser-2.3.4/include/muParserIncluder.h"

#include "../include/project/inputmanager.hpp"
#include "../include/project/hypersphere.hpp"
#include "../include/project/hyperrectangle.hpp"
#include "../include/project/hypercube.hpp"
#include "../include/project/asset.hpp"
#include "../include/project/montecarlo.hpp"
#include "../include/project/finance_inputmanager.hpp"
#include "../include/project/finance_montecarlo.hpp"
#include "../include/project/optionparameters.hpp"
#include "../include/project/functionevaluator.hpp"
#include "../include/project/finance_computation.hpp"
#include "../include/project/integrationcomputation.hpp"

int main()
{
    int choice;
    bool validChoice = false;

    while (!validChoice)
    {
        std::cout << "Choose computation type:" << std::endl;
        std::cout << "1. Finance Monte Carlo" << std::endl;
        std::cout << "2. Monte Carlo Integration" << std::endl;
        std::cin >> choice;

        if (std::cin.fail())
        {
            std::cin.clear ();                                                   // Clear the fail state
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Clear the input buffer
            std::cout << "Invalid input. Please enter a number." << std::endl;
        }
        else
        {
            switch (choice)
            {
            case 1: 
                financeComputation();
                validChoice = true;
                break;
            case 2: 
                integrationComputation();
                validChoice = true;
                break;
            default: 
                std::cout << "Invalid choice. Please enter 1 or 2." << std::endl;
                break;
            }
        }
    }
    return 0;
}
