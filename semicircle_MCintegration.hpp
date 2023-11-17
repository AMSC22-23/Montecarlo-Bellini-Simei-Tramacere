#include <random>
#include <iostream>
#include <cmath>
#include <chrono>


int MC_integration(int dim) {

        int n=1000000;
        double radius=21.0;

        double x[2];
        double omega,integral;
        double sum=0;

        std::random_device rd;
        auto seed = rd();

        std::default_random_engine eng(seed);

        auto start = std::chrono::high_resolution_clock::now(); // start the timer

        for( int i = 0; i < n; ++i ) {
                std::uniform_real_distribution<double> distribution(0.0,radius);
                x[0] = distribution(eng);
                x[1] = distribution(eng);
                if ( pow(x[0],2) + pow(x[1],2) <= pow(radius,2) ) {
                        sum += 1;
                }
        }

        omega = 2.0*pow(radius,2);
        integral = (omega/static_cast<double>(n))*sum;
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "The approximate result of your integral is: " << integral << std::endl;

        auto end = std::chrono::high_resolution_clock::now(); // stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // calculate the duration
        std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl; // print the duration

        return 0;
}