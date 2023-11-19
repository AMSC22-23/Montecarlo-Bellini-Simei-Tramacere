#include <iostream>
#include <string>

#include "nD_sphere_MCintegration.hpp"

int main()
{

    std::vector<int> dim(6);
    dim = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++)
    {
        nD_sphere_MC_integration(dim[i]);
    }
    return 0;
}