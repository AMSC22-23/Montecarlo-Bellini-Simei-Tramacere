#include <iostream>
#include <string>
#include <cmath>

#include "nD_sphere_MCintegration.hpp"
#include "sphere_volume.hpp"

int main()
{

    std::vector<int> dim(6);
    dim = {1, 2, 3, 4, 5, 6};
    std::pair<double, double> result;
    double volume;

    for (int i = 0; i < 6; i++)
    {
        result = nD_sphere_MC_integration(dim[i]);
        std::cout << "The approximate result in " << dim[i] << " dimensions of your integral is: " << result.first << std::endl;
        std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;
        volume = sphere_volume(dim[i], 1.0);
        std::cout << "The exact result in " << dim[i] << " dimensions of your integral is: " << volume << std::endl;
        std::cout << "The absolute error is: " << std::abs(result.first - volume) << std::endl;
        std::cout << "The relative error is: " << std::abs(result.first - volume) / volume << std::endl << std::endl;
    }

    return 0;
}