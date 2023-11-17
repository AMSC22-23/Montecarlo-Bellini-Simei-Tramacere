#include <iostream>
#include <string>

#include "semicircle_MCintegration.hpp"


int main() {

    int dim;

    std::cout << "In how many dimensions your domain of integration is?" << std::endl;
    std::cin >> dim;

    MC_integration(dim);

    return 0;
}