#include <iostream>
#include <string>
#include "semicircle_MCintegration.hpp"


int main() {

    int n;
    //int dim;
    double radius=21.0;

    std::cout << "How many points do you want to generate?" << std::endl;
    std::cin >> n;
    //std::cout << "In how many dimensions your domain of integration is?" << std::endl;
    //std::cin >> dim;

    MC_integration(n,radius);

    return 0;
}