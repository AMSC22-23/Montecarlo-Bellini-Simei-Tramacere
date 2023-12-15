#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "HyperSphere.cpp"


// note that the following method implements rn only the special case with the function f(x) = 1
// TODO: generalize the method for all the functions

std::pair<double, double> Montecarlo_integration(HyperSphere& hypersphere, int n) {
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // generate the random points
    for (int i = 0; i < n; ++i) {
        std::vector<double> random_point = hypersphere.generate_random_point();
        if (!random_point.empty()) hypersphere.add_point_inside();
    }
    // calculate the integral
    hypersphere.calculate_approximated_volume(n);
    double integral = hypersphere.get_approximated_volume();
    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::make_pair(integral, duration.count());
}


int main(int argc, char** argv) {
    int n, dim;
    double rad;
    // ask the user to insert the parameters
    std::cout << "Insert the number of random points to generate: ";
    std::cin >> n;
    std::cout << "Insert the dimension of the hypersphere: ";
    std::cin >> dim;
    std::cout << "Insert the radius of the hypersphere: ";
    std::cin >> rad;
    // generate the hypersphere
    HyperSphere hypersphere(dim, rad);
    // calculate the integral and print the results
    std::pair<double, double> result = Montecarlo_integration(hypersphere, n);
    std::cout << std::endl << "The approximate result in " << dim << " dimensions of your integral is: " << result.first << std::endl;
    std::cout << "The time needed to calculate the integral is: " << result.second << " microseconds" << std::endl;
    hypersphere.calculate_volume();
    double volume = hypersphere.get_volume();
    std::cout << "The exact result in " << dim << " dimensions of your integral is: " << volume << std::endl;
    std::cout << "The absolute error is: " << std::abs(result.first - volume) << std::endl;
    std::cout << "The relative error is: " << std::abs(result.first - volume) / volume << std::endl;
    return 0;
}