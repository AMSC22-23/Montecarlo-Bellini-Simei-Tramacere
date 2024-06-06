#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <fstream>

#include "../external/muparser-2.3.4/include/muParser.h"
#include "../external/muparser-2.3.4/include/muParserIncluder.h"

#include "../include/integration/integralcalculator.hpp"
#include "../include/optionpricing/optionpricer.hpp"

  // Main function
int main()
{
  int choice;
  bool validChoice = false;

  while (!validChoice)
  {
      // Display the menu
    std::cout << "What you want to do?\n1. Price an option\n2. Calculate an integral\nEnter choice (1 or 2): ";

    std::cin >> choice;

    if (std::cin.fail())
    {
        // Clear the fail state
      std::cin.clear();
        // Clear the input buffer
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      std::cout << "Invalid input. Please enter a number." << std::endl;
    }
    else
    {
        // Check the choice
      switch (choice)
      {
      case 1: 
          // Call the finance computation function
        financeComputation();
        validChoice = true;
        break;
      case 2: 
          // Call the integral calculator function
        integralCalculator();
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