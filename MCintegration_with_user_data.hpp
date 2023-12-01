#include <random>
#include <iostream>
#include <cmath>

// TODO generalizzare in modo che accetti una funzione in input decisa dall'utente
int MC_integration(int n, double domain_bound, int dim) {

        std::vector<double> x(dim);
        double integral;
        double sum=0;
        long double domain_check=0;

        std::random_device rd;

        std::default_random_engine eng( rd() );
        for( int i = 0; i < n; ++i ) {

                std::uniform_real_distribution<double> distribution(-domain_bound,domain_bound);

                for( int j = 0; j < dim; ++j ) x.emplace_back(distribution(eng));

                for( auto & j : x ) domain_check += pow(j,2);

                for( auto & j : x ) sum += pow(j,2);

                //CALCOLA VOLUME SFERA
                /*if ( domain_check <= pow(domain_bound,2) ) {  // if the point is inside the (hyper)sphere
                        for( int j=0; j < dim; ++j ) {  
                                sum += 1; // calculate the (hyper)volume
                        }
                }*/
                domain_check = 0;
        }

        double domain_size = 1.;
        for( int j = 0; j < dim; ++j ) domain_size *= 2. *domain_bound; // calculate the (hyper)volume

        integral = domain_size*sum/(double)n; //TODO generalizzare il dominio per ogni dimensione
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "The approximate result of your integral is: " << integral << std::endl;

        return 0;
}