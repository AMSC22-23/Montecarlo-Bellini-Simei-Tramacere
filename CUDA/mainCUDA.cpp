#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "../include/optionpricing/asset.hpp"
#include "../include/optionpricing/optionpricer.hpp"
#include "../include/optionpricing/finance_montecarlo.hpp"
#include "../include/optionpricing/optionparameters.hpp"
#include "../include/optionpricing/finance_inputmanager.hpp"

  // Extern function declaration for the kernel_wrapper function
extern std::pair<double, double> kernel_wrapper(long long int N, const std::string &function,
                                                const std::vector<const Asset *> &assetPtrs, double *variance,
                                                std::vector<double> coefficients, double strike_price,
                                                OptionType option_type);

  // Function to get the option type from the user
OptionType cuda_getOptionTypeFromUser()
{
    int        input  = 0;
    OptionType option = OptionType::Invalid;

    std::cout << "\nSelect the option type:\n1. European\n2. Asian\nEnter choice (1 or 2): ";

    while (true)
    {
        std::cin >> input;

        if (std::cin.fail() || (input != 1 && input != 2))
        {
            std::cin.clear ();                                                   // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard invalid input
            std::cout << "\nInvalid input. Please enter 1 for European or 2 for Asian." << std::endl;
        }
        else
        {
            option = static_cast<OptionType>(input);
            break;
        }
    }

    return option;
}

  // Function to get the asset count type from the user
AssetCountType cuda_getAssetCountTypeFromUser()
{
    int            input          = 0;
    AssetCountType assetCountType = AssetCountType::Invalid;

    std::cout << "\nSelect the asset count type:\n1. Single\n2. Multiple\nEnter choice (1 or 2): ";

    while (true)
    {
        std::cin >> input;

        if (std::cin.fail() || (input != 1 && input != 2))
        {
            std::cin.clear ();                                                   // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard invalid input
            std::cout << "\nInvalid input. Please enter 1 for Single or 2 for Multiple." << std::endl;
        }
        else
        {
            assetCountType = static_cast<AssetCountType>(input);
            break;
        }
    }

    return assetCountType;
}

int main(int argc, char **argv)
{
    using namespace std::chrono;

      // Initialize the number of points for Monte Carlo simulation
    long   long int N     = 1e6;
    double strike_price   = 0.0;
    double variance       = 0.0;
    size_t num_iterations = 10;   // Number of iterations for the simulation

    std::vector<Asset> assets;  // Vector to store asset objects

      // Get the option type from the user
    OptionType option_type = cuda_getOptionTypeFromUser();
    if (option_type == OptionType::Invalid)
    {
        std::cerr << "\nInvalid option type" << std::endl;
        exit(1);  // Exit the program if the option type is invalid
    }

      // If the option type is Asian, set the number of points to 1e6
    if (option_type == OptionType::Asian)
    {
        N = 1e6;
    }

      // Get the asset count type from the user
    AssetCountType asset_count_type = cuda_getAssetCountTypeFromUser();
    if (asset_count_type == AssetCountType::Invalid)
    {
        std::cerr << "\nInvalid asset count type" << std::endl;
        exit(1);  // Exit the program if the asset count type is invalid
    }

      // Load the assets from the CSV files
    std::cout << "\nLoading assets from csv..." << std::endl;

    LoadAssetError load_result = loadAssets("../../data/", assets, asset_count_type);
    switch (load_result)
    {
    case LoadAssetError::Success: 
        std::cout << "The assets have been loaded successfully.\n"
                  << std::endl;
        break;
    case LoadAssetError::DirectoryOpenError: 
        exit(1);  // Exit the program if there is a directory open error
    case LoadAssetError::NoValidFiles: 
        std::cout << "No valid files found in the directory\n"
                  << std::endl;
        exit(1);  // Exit the program if there are no valid files
    case LoadAssetError::FileReadError: 
        exit(1);  // Exit the program if there is a file read error
    }

      // Create a vector of pointers to the asset objects
    std::vector<const Asset *> assetPtrs;
    assetPtrs.reserve(assets.size());
    for (const auto &asset : assets)
    {
        assetPtrs.emplace_back(&asset);
    }

    printf("Pricing the option ...\n");
      // Get the starting timepoint for measuring execution time
    auto start        = high_resolution_clock::now();
         strike_price = calculateStrikePrice(assets);  // Calculate the strike price

      // Create the payoff function and get the coefficients
    std::pair<std::string, std::vector<double>> function_pair = createPayoffFunction(strike_price, assets);
    auto                  function                            = function_pair.first;
    auto                  coefficients                        = function_pair.second;

    std::pair<double, double> result;
    std::pair<double, double> result_temp;
    result.first  = 0.0;
    result.second = 0.0;

      // Perform the Monte Carlo simulation for the given number of iterations
    for (size_t i = 0; i < num_iterations; ++i)
    {
        result_temp = kernel_wrapper(N, function, assetPtrs, &variance,
                                     coefficients, strike_price, option_type);
        result.first  += result_temp.first;
        result.second += result_temp.second;
    }
    result.first /= num_iterations;  // Calculate the average final price
    variance     /= num_iterations;  // Calculate the average variance

      // Open the output file stream
    std::ofstream outputFile("output.txt", std::ios::app);

      // Check if the file is empty
    outputFile.seekp(0, std::ios::end);
    bool isEmpty = (outputFile.tellp() == 0);
    outputFile.seekp(0, std::ios::beg);

      // Write the title of the columns if the file is empty
    if (isEmpty)
    {
          // Print asset information to the file in a table format
        outputFile << std::left << std::setw(20) << "Asset";
        outputFile << std::left << std::setw(20) << "Return Mean";
        outputFile << std::left << std::setw(20) << "Return Standard Deviation\n";

        for (const auto &asset : assets)
        {
            outputFile << std::left << std::setw(20) << asset.getName();
            outputFile << std::left << std::setw(20) << asset.getReturnMean();
            outputFile << std::left << std::setw(20) << asset.getReturnStdDev() << "\n";
        }
        outputFile << "=============================================================================================\n";

          // Write additional information to the file
        outputFile << "The function is: " << function << "\n";

        outputFile << "=============================================================================================\n";

        outputFile << std::left << std::setw(15) << "Points";
        outputFile << std::left << std::setw(15) << "Variance";
        outputFile << std::left << std::setw(15) << "Final Price";
        outputFile << std::left << std::setw(15) << "Time"
                   << "\n";
    }

      // Write important data to the file with increased column width
    outputFile << std::left << std::setw(15) << N;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << variance;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.first;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.second * 1e-6 << "\n";

      // Close the output file stream
    outputFile.close();

      // Get the ending timepoint for measuring execution time
    auto stop = high_resolution_clock::now();
      // Get the duration by subtracting the start and stop timepoints
      // and casting the result to milliseconds
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " milliseconds" << std::endl;

    printf("Done\n");
    printf("\n");
    return 0;
}