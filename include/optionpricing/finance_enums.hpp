/**
 * @file finance_enums.hpp
 * @brief This file contains enumerations related to finance.
 */

#ifndef FINANCE_ENUMS_HPP
    #define FINANCE_ENUMS_HPP

// Enum for asset loading errors
enum class LoadAssetError
{
    Success,           /**< Indicates successful asset loading */
    DirectoryOpenError,/**< Error opening the directory */
    NoValidFiles,      /**< No valid files found in the directory */
    FileReadError      /**< Error reading the file */
};

// Enum for option types
enum class OptionType {
    European = 1,
    Asian,
    Invalid
};

// Enum for asset count types
enum class AssetCountType {
    Single = 1,
    Multiple,
    Invalid
};

// Enum for covariance calculation errors
enum class CovarianceError {
    Success, /**< Indicates successful covariance calculation */
    Failure  /**< Indicates failure in covariance calculation */
};

// Enum for Monte Carlo simulation errors
enum class MonteCarloError {
    Success,              /**< Indicates successful Monte Carlo simulation */
    PointGenerationFailed /**< Indicates failure in generating random points for Monte Carlo simulation */
};

#endif
