#include <random>
#include <iostream>
#include <cmath>
#include <chrono>


std::pair<double,double> nD_sphere_MC_integration(int dim)
{

        int n=10000;                 // number of points
        double radius=1., volume=1.; // initial volume of the hypercube (length of the side)

        for( int i = 0; i < dim; ++i ) volume *= 2. * radius;

        // std::cout << "volume is " << volume << std::endl;
        auto start = std::chrono::high_resolution_clock::now(); // start the timer

        std::random_device rd; // obtain a random number from hardware

        std::default_random_engine eng(rd()); // seed the generator

        std::uniform_real_distribution<double> distribution(-radius, radius); // define the range

        int points_inside = 0; // number of points inside the semicircle

        std::vector<double> x;       // vector of random numbers
        x.reserve(static_cast<std::size_t>(dim)); // reserve the memory for the vector
        double sum = 0.;             // sum of the random numbers
        for( int i = 0; i < n; ++i )
        {
                for( int j = 0; j < dim; ++j ) x.emplace_back(distribution(eng));

                sum = 0; // reset the sum
                for( int j = 0; j < x.size(); ++j ) sum += pow(x[j], 2);

                if( sum <= pow(radius, 2) ) ++points_inside;
        }
        // std::cout << "points_inside is " << points_inside << std::endl;

        double ratio = static_cast<double>(points_inside) / n; // ratio between the points inside the semicircle and the total number of points
        double integral = ratio * volume;                      // approximate value of the integral

        auto end = std::chrono::high_resolution_clock::now();                                 // stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);   // calculate the duration

        auto result = std::make_pair(integral, duration.count()); // return the integral and the duration

        return result;

}