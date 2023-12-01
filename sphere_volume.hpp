#include <iostream>
#include <string>
#include <cmath>


double sphere_volume(int dim, int radius) 
{

    double param, volume;
    param = dim / 2.;

    volume = pow(M_PI, param) / tgamma(param + 1.) * pow(radius, dim);

    return volume;
}