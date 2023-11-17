#include <random>
#include <iostream>
#include <cmath>


int MC_integration(int n, double radius) {

        //Monte Carlo integration of f=1 over a semicircle of radius "radius"

        int i;
        double x,y;//coordinates of points
        double integral;//approximation of the integral
        double sum =0; //value of function at points inside circle

        std::default_random_engine generator{124321};
        for( i = 0; i < n; i++) {
                std::uniform_real_distribution<double> distribution(0.0,sqrt(radius));
                x = distribution(generator);
                y = distribution(generator);
                if( x*x + y*y <= radius) {
                        sum += 1;
                }
        }

        integral = (2.0*radius*radius/(double)n)*(sum);
        std::cout << "area is approximately: " << integral << std::endl;

        return 0;
}