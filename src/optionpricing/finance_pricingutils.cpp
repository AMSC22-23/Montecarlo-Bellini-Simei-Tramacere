#include "../../include/optionpricing/finance_pricingutils.hpp"

  // Function that calculates the value of the standard normal distribution function
double phi(const double x)
{
    constexpr double RT2PI = 2.50662827463100;
    constexpr double SPLIT = 7.07106781186547;
    constexpr double N0    = 220.206867912376;
    constexpr double N1    = 221.213596169931;
    constexpr double N2    = 112.079291497871;
    constexpr double N3    = 33.912866078383;
    constexpr double N4    = 6.37396220353165;
    constexpr double N5    = 0.700383064443688;
    constexpr double N6    = 3.52624965998911e-02;
    constexpr double M0    = 440.413735824752;
    constexpr double M1    = 793.826512519948;
    constexpr double M2    = 637.333633378831;
    constexpr double M3    = 296.564248779674;
    constexpr double M4    = 86.7807322029461;
    constexpr double M5    = 16.064177579207;
    constexpr double M6    = 1.75566716318264;
    constexpr double M7    = 8.83883476483184e-02;

    const  double z = fabs(x);
    double c        = 0.0;

    if (z <= 37.0)
    {
        const double e = exp(-z * z / 2.0);
        if (z < SPLIT)
        {
            const double n = (((((N6 * z + N5) * z + N4) * z + N3) * z + N2) * z + N1) * z + N0;
            const double d = ((((((M7 * z + M6) * z + M5) * z + M4) * z + M3) * z + M2) * z + M1) * z + M0;
                  c        = e * n / d;
        }
        else
        {
            const double f = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 13.0 / 20.0))));
                  c        = e / (RT2PI * f);
        }
    }
    return x <= 0.0 ? c : 1 - c;
}

  // Function that writes the results of the option pricing to a file
void writeResultsToFile(const std::vector<Asset> &assets,
                        const std::pair<double, double> &result,
                        const double &standard_error,
                        const std::string &function,
                        const size_t &num_simulations,
                        const OptionType &option_type)
{
    std::ofstream outputFile("output.txt", std::ios::app);
    outputFile.seekp(0, std::ios::end);
    bool isEmpty = (outputFile.tellp() == 0);
    outputFile.seekp(0, std::ios::beg);
    std::string option_type_s = (option_type == OptionType::European) ? "European" : "Asian";

    if (isEmpty)
    {
          // Get current time and date
        auto        now  = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

          // Convert time to string
        std::string timeStr = std::ctime(&time);
        timeStr.pop_back();  // Remove the newline character at the end

          // Parse the date and time string
        std::istringstream ss(timeStr);
        std::string dayOfWeek, month, day, timeOfDay, year;
        ss >> dayOfWeek >> month >> day >> timeOfDay >> year;

          // Reformat the string to have the year before the time
        std::ostringstream formattedTimeStr;
        formattedTimeStr << dayOfWeek << " " << month << " " << day << " " << year << " " << timeOfDay;

          // Write the results to output.txt
        outputFile << "Generated on: " << formattedTimeStr.str() << "\n";
        outputFile << "==================================================================================================================================\n";
        outputFile << std::left << std::setw(22) << "Asset";
        outputFile << std::left << std::setw(22) << "Return Mean";
        outputFile << std::left << std::setw(22) << "Return Standard Deviation\n";
        for (const auto &asset : assets)
        {
            outputFile << std::left << std::setw(22) << asset.getName();
            outputFile << std::left << std::setw(22) << asset.getReturnMean();
            outputFile << std::left << std::setw(22) << asset.getReturnStdDev() << "\n";
        }
        outputFile << "==================================================================================================================================\n";
        outputFile << "The function is: " << function << "\n";
        outputFile << "==================================================================================================================================\n";
        outputFile << std::left << std::setw(22) << "Points";
        outputFile << std::left << std::setw(22) << "Standard error";
        outputFile << std::left << std::setw(22) << "Option payoff";
        outputFile << std::left << std::setw(22) << "Time[s]";
        outputFile << std::left << std::setw(22) << "Option type" << "\n";
    }

      // Write the results to output.txt
    outputFile << std::left << std::setw(22) << num_simulations;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << standard_error;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.first;
    outputFile << std::fixed << std::setprecision(6) << std::left << std::setw(22) << result.second * 1e-6 ;
    outputFile << std::left << std::setw(22) << option_type_s << "\n";
    outputFile.close();
}

  // Function that computes the Black-Scholes option price
double computeBlackScholesOptionPrice(const std::vector<const Asset *> &assetPtrs, const double &strike_price)
{
    double S     = 0.0;           // Stock price
    double r     = 0.05;          // Risk-free rate
    double sigma = 0.0;           // Volatility
    double T     = 1;             // Time to maturity
    double K     = strike_price;  // Strike price

    for (size_t i = 0; i < assetPtrs.size(); ++i)
    {
        S     += assetPtrs[i]->getLastRealValue();
        sigma += assetPtrs[i]->getReturnStdDev();
    }
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S * phi(d1) - K * exp(-r * T) * phi(d2);
}

  // Function to get user input and validate
OptionType getOptionTypeFromUser()
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

  // Function to get user input for asset count type and validate
AssetCountType getAssetCountTypeFromUser()
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
