#include <iostream>
#include <string>
#include "MCintegration_with_user_data.hpp"
#include "nD_sphere_MCintegration.hpp"
#include "sphere_volume.hpp"



int main() {
    
    // Test cases
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

    // User input (more dynamic code)
    int n; // TODO fare una ENUM con LOW, MEDIUM, HIGH per il numero di punti, che indicano l'accuratezza dell'integrale
    double domain_bound;
    int user_given_dim;

    std::cout << "What is the upper bound of the integration domain? (note that the domain of integration is symmetric)" << std::endl;
    std::cin >> domain_bound;
    std::cout << "How many points do you want to generate?" << std::endl;
    std::cin >> n;
    std::cout << "In how many dimensions your domain of integration is?" << std::endl;
    std::cin >> user_given_dim;

    MC_integration(n,domain_bound, user_given_dim);

    return 0;
}