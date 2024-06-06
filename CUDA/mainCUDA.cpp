

#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "../include/optionpricing/asset.hpp"
#include "../include/optionpricing/optionpricer.hpp"
#include "../include/optionpricing/finance_montecarlo.hpp"
#include "../include/optionpricing/optionparameters.hpp"
#include "../include/optionpricing/finance_inputmanager.hpp"

extern std::pair<double, double> kernel_wrapper(long long int N, const std::string &function,
                                                const std::vector<const Asset *> &assetPtrs , double *variance ,
                                                std::vector<double> coefficients, double strike_price,
                                                OptionType option_type);

OptionType cuda_getOptionTypeFromUser()
{
    int input = 0;
    OptionType option = OptionType::Invalid;

    std::cout << "\nSelect the option type:\n1. European\n2. Asian\nEnter choice (1 or 2): ";

    while (true)
    {
        std::cin >> input;

        if (std::cin.fail() || (input != 1 && input != 2))
        {
            std::cin.clear();                                                   // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
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

// Function to get user input for asset count type and validate
AssetCountType cuda_getAssetCountTypeFromUser()
{
    int input = 0;
    AssetCountType assetCountType = AssetCountType::Invalid;

    std::cout << "\nSelect the asset count type:\n1. Single\n2. Multiple\nEnter choice (1 or 2): ";

    while (true)
    {
        std::cin >> input;

        if (std::cin.fail() || (input != 1 && input != 2))
        {
            std::cin.clear();                                                   // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
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

    // long long int N = 4194304;
    // long long int N = 1e8;
    long long int N = 1e6;
    double strike_price = 0.0;
    double variance = 0.0;
    size_t num_iterations = 10;

    std::vector<Asset> assets;

    OptionType option_type = cuda_getOptionTypeFromUser();
    if (option_type == OptionType::Invalid)
    {
        std::cerr << "\nInvalid option type" << std::endl;
        exit(1);
    }


    if (option_type == OptionType::Asian)
    {
        N = 1e6;
    }

    AssetCountType asset_count_type = cuda_getAssetCountTypeFromUser();
    if (asset_count_type == AssetCountType::Invalid)
    {
        std::cerr << "\nInvalid asset count type" << std::endl;
        exit(1);
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
        exit(1);
    case LoadAssetError::NoValidFiles:
        std::cout << "No valid files found in the directory\n"
                  << std::endl;
        exit(1);
    case LoadAssetError::FileReadError:
        exit(1);
    }

    std::vector<const Asset *> assetPtrs;
    assetPtrs.reserve(assets.size());
    for (const auto &asset : assets)
    {
        assetPtrs.emplace_back(&asset);
    }

    printf("Pricing the option ...\n");
    // Get starting timepoint
    auto start = high_resolution_clock::now();
    strike_price = calculateStrikePrice(assets);

    // std::string function = create_function(strike_price, assets);
    std::pair<std::string, std::vector<double>> function_pair = createPayoffFunction(strike_price, assets);
    auto function = function_pair.first;
    auto coefficients = function_pair.second;


   
    std::pair<double, double> result;
    std::pair<double, double> result_temp;
    result.first = 0.0;
    result.second = 0.0;

    for (size_t i = 0; i < num_iterations; ++i)
    {
        result_temp = kernel_wrapper(N, function, assetPtrs, &variance,
                                     coefficients, strike_price, option_type);
        result.first += result_temp.first;
        result.second += result_temp.second;
    }
    result.first /= num_iterations;
    variance /= num_iterations;

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

    // Write important data to file with increased column width
    outputFile << std::left << std::setw(15) << N;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << variance;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.first;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(15) << result.second * 1e-6 << "\n";


    // Close the output file stream
    outputFile.close();

    auto stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " milliseconds" << std::endl;

    printf("Done\n");
    printf("\n");
    return 0;
}