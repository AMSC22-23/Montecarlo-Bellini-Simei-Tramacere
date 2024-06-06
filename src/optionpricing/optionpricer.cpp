#include "../../include/optionpricing/optionpricer.hpp"

  // Function that embeds multiple methods used to compute
  // the option price using the Monte Carlo method
void financeComputation()
{
      // Vector to store assets
    std::vector<Asset> assets;

      // Get option type from user input
    OptionType option_type = getOptionTypeFromUser();
    if (option_type == OptionType::Invalid)
    {
        std::cerr << "\nInvalid option type" << std::endl;
        exit(1);
    }

      // Get asset count type from user input
    AssetCountType asset_count_type = getAssetCountTypeFromUser();
    if (asset_count_type == AssetCountType::Invalid)
    {
        std::cerr << "\nInvalid asset count type" << std::endl;
        exit(1);
    }

      // Load the assets from the CSV files
    std::cout << "\nLoading assets from csv..." << std::endl;

    LoadAssetError load_result = loadAssets("../data/", assets, asset_count_type);
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

      // Create a vector of pointers to assets for Monte Carlo computation
    std::vector<const Asset *> assetPtrs;
    assetPtrs.reserve(assets.size());
    for (const auto &asset : assets)
    {
        assetPtrs.emplace_back(&asset);
    }

      // Set the number of iterations and simulations based on the option type
    size_t num_iterations  = 10;
    size_t num_simulations = (option_type == OptionType::European) ? 1e6 : 1e5;
    double strike_price    = calculateStrikePrice(assets);
    double variance        = 0.0;
    double variance_temp   = 0.0;
    double standard_error  = 0.0;
    std::pair<double, double> result;
    std::pair<double, double> result_temp;
    result.first  = 0.0;
    result.second = 0.0;
    MonteCarloError error;

      // Create the payoff function and coefficients
    auto function_pair = createPayoffFunction(strike_price, assets);
    auto function      = function_pair.first;
    auto coefficients  = function_pair.second;

      // Vector to store predicted asset prices
    std::vector<double> predicted_assets_prices;
    predicted_assets_prices.resize(assets.size());

    std::cout << "Calculating the price of the option...\n"
              << std::endl;

      // Apply the Monte Carlo method to calculate the price of the option
    for (size_t j = 0; j < num_iterations; ++j)
    {
        result_temp = monteCarloPricePrediction(num_simulations,
                                                assetPtrs,
                                                variance_temp,
                                                strike_price,
                                                predicted_assets_prices,
                                                option_type,
                                                error);

        if (error != MonteCarloError::Success)
        {
            std::cerr << "Error in Monte Carlo simulation" << std::endl;
            exit(1);
        }

        result.first   += result_temp.first;
        result.second  += result_temp.second;
        variance       += variance_temp;
        standard_error += std::sqrt(variance_temp / static_cast<double>(num_simulations));

        double progress = static_cast<double>(j + 1) / static_cast<double>(num_iterations) * 100;
        std::cout << "Process at " << progress << "% ..." << std::endl;
    }

      // Calculate averages
    result.first   /= num_iterations;
    variance       /= num_iterations;
    standard_error /= num_iterations;

      // Normalize predicted asset prices
    for (size_t i = 0; i < assetPtrs.size(); ++i)
    {
        predicted_assets_prices[i] /= (num_iterations * num_simulations);
    }

      // Output option price calculated via Black-Scholes model if applicable
    std::cout << std::endl;
    if (option_type == OptionType::European && asset_count_type == AssetCountType::Single)
    {
        double BS_option_price = computeBlackScholesOptionPrice(assetPtrs, strike_price);
        std::cout << "The option expected payoff calculated via Black-Scholes model is " << BS_option_price << std::endl;
    }

      // Output option price calculated via Monte Carlo method
    std::cout << "The option expected payoff calculated via Monte Carlo method is " << result.first << std::endl;

      // Write results to file
    writeResultsToFile(assets, result, standard_error, function, num_simulations, option_type);

      // Output information about the calculation
    std::cout << "\nThe integral has been calculated successfully " << num_iterations << " times for " << num_simulations << " points." << std::endl;
    std::cout << "The resulting expected discounted option payoff is the average of the " << num_iterations << " iterations.\n";
    std::cout << "\nThe results have been saved to output.txt\n"
              << std::endl;

      // Output predicted future prices of assets
    for (size_t i = 0; i < assets.size(); ++i)
    {
        std::cout << "The predicted future prices (one year) of one " << assets[i].getName() << " stock is " << predicted_assets_prices[i] << std::endl;
    }
}
