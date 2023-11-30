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
                for( int j = 0; j < dim; ++j ) { //random assignment of the sample points
                        x[j] = distribution(eng);
                }


                for( int j=0; j < dim; ++j ) {          // check if the point is inside the (hyper)sphere
                        domain_check += pow(x[j],2);
                        //std::cout << "domain_check:" << domain_check << std::endl;
                }

                for( int j=0; j < dim; ++j ) {  
                                //sum += class.my_function( pow(x[j],2) );// TODO generalizzare la funzione
                                sum +=  pow(x[j],2);
                                
                        }


                //CALCOLA VOLUME SFERA
                /*if ( domain_check <= pow(domain_bound,2) ) {  // if the point is inside the (hyper)sphere
                        for( int j=0; j < dim; ++j ) {  
                                sum += 1; // calculate the (hyper)volume
                        }
                }*/
                domain_check = 0;
        }

        double domain_size = 1.0;
        for( int j=0; j < dim; ++j ) {  
                domain_size *= 2.0*domain_bound; // calculate the (hyper)volume
        }

        integral = domain_size*sum/(double)n; //TODO generalizzare il dominio per ogni dimensione
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "The approximate result of your integral is: " << integral << std::endl;

        return 0;
}