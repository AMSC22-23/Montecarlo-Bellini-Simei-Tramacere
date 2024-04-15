#ifndef PROJECT_CSVHANDLER_
#define PROJECT_CSVHANDLER_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <omp.h>
#include <cmath>


#include "asset.hpp"

int read_csv(const std::string& filename, Asset* asset_ptr);

int load_assets_from_csv(const std::string& directory, std::vector<Asset>& assets);

#endif