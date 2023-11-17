#include <random>
#include <iostream>
#include <cmath>
#include <chrono>


int MC_integration(int n, double radius) {

        double x, y;
        double integral;
        double sum=0;

        std::random_device rd;

        std::default_random_engine eng( rd() );

        auto start = std::chrono::high_resolution_clock::now(); // start the timer

        for( int i = 0; i < n; ++i ) {
                std::uniform_real_distribution<double> distribution(0.0,radius);
                x = distribution(eng);
                y = distribution(eng);
                if ( pow(x,2) + pow(y,2) <= pow(radius,2) ) {
                        sum += 1;
                }
        }

        integral = (2.0*radius*radius/(double)n)*(sum);
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "The approximate result of your integral is: " << integral << std::endl;

        auto end = std::chrono::high_resolution_clock::now(); // stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // calculate the duration
        std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl; // print the duration

        return 0;
}