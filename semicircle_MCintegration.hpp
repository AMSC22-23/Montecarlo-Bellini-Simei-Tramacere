#include <random>
#include <iostream>
#include <cmath>
#include <chrono>


int MC_integration(int dim) {

        int n=1000000;// number of points
        double radius=1, volume=1;// initial volume of the hypercube (length of the side)

        for (int i=0; i<dim; i++) {// volume of the hypercube
                volume*=2*radius;
        }
        //std::cout << "volume is " << volume << std::endl;
        auto start = std::chrono::high_resolution_clock::now(); // start the timer

        std::random_device rd;// obtain a random number from hardware
        
        std::default_random_engine eng(rd());// seed the generator

        std::uniform_real_distribution<double> distribution(-radius,radius);// define the range

        int points_inside=0;// number of points inside the semicircle

        std::vector<double> x(dim);// vector of random numbers
        double sum=0;// sum of the random numbers
        for (int i=0; i<n; i++) {

                for (int j=0; j<dim; j++) { // fill the vector
                        x[j]=distribution(eng);// generate the random numbers
                }

                sum=0;
                for (int j=0; j<dim; j++) {
                        sum+=pow(x[j],2);
                }

                if (sum<=pow(radius,2)) {// check if the point is inside the semicircle
                        points_inside++;
                }
        }
        //std::cout << "points_inside is " << points_inside << std::endl;


        double ratio= static_cast<double>(points_inside)/n;// ratio between the points inside the semicircle and the total number of points
        double integral=ratio*volume;// approximate value of the integral
        
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "The approximate result in " << dim << " dimensions of your integral is: " << integral << std::endl;

        auto end = std::chrono::high_resolution_clock::now(); // stop the timer
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // calculate the duration
        std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl; // print the duration

        return 0;
}