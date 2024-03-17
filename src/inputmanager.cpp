#include "../include/project/inputmanager.hpp"
#include <iostream>
#include <chrono>
#include <vector>



// Function to manage the input of the user
void input_manager(int &n, int &dim, double &rad, double &edge, std::string &function, std::string &domain_type, std::vector<double> &hyper_rectangle_bounds) {
    

    // Choose the type of domain to integrate
    std::cout << "Insert the type of domain you want to integrate: (hc for hyper-cube, hs for hyper-sphere, hr for hyper-rectangle)" << std::endl;
    std::cin >> domain_type;
    while (std::cin.fail() || (domain_type != "hs" && domain_type != "hr" && domain_type != "hc")) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter hs for hyper-sphere, hr for hyper-rectangle: ";
        std::cin >> domain_type;
    }

    // Get and validate number of random points
    std::cout << "Insert the number of random points to generate: ";
    std::cin >> n;
    while (std::cin.fail() || n <= 0) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Please enter a positive integer: ";
        std::cin >> n;
    }


    // If the domain is a hyper-sphere, ask for the radius and dimension
    if (domain_type == "hs") {
        std::cout << "Insert the dimension of the hypersphere: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }
        std::cout << "Insert the radius of the hypersphere: ";
        std::cin >> rad;
        while (std::cin.fail() || rad <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> rad;
        }
    }

    // If the domain is a hyper-rectangle, ask for the dimension and the bounds
    else if (domain_type == "hr") {
        std::cout << "Insert the dimension of the hyperrectangle: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }
        hyper_rectangle_bounds.reserve(dim*2);
        double tmp;
        for (int i = 0; i < 2 * dim; i++) {
            if (i == 0)
                std::cout << "Insert the 1st dimension coordinate of the hyper-rectangle: ";
            else if (i == 1)
                std::cout << "Insert the 2nd dimension coordinate of the hyper-rectangle: ";
            else if (i == 2)
                std::cout << "Insert the 3rd dimension coordinate of the hyper-rectangle: ";
            else
                std::cout << "Insert the " << i+1 << "th dimension coordinate of the hyper-rectangle: ";
            std::cin >> tmp;
            hyper_rectangle_bounds.push_back(tmp);
            while (std::cin.fail()) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a positive number: ";
                std::cin >> hyper_rectangle_bounds[i];
            }
        }
    }

    // If the domain is a hyper-cube, ask for the dimension and the edge length
    else if (domain_type == "hc") {
        std::cout << "Insert the dimension of the hypercube: ";
        std::cin >> dim;
        while (std::cin.fail() || dim <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive integer: ";
            std::cin >> dim;
        }
        std::cout << "Insert the edge length of the hypercube: ";
        std::cin >> edge;
        while (std::cin.fail() || edge <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a positive number: ";
            std::cin >> rad;
        }
    }


    // Ask the user to insert the function to integrate
    std::cout << "Insert the function to integrate: ";
    std::cin >> function;
}



// Function to manage the input file
int csv_reader(const std::string& filename, Asset* asset_ptr) {
    
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open())
        return -1;

    std::string line;
    // Skip the header line
    std::getline(file, line);

    double total_return_percentage = 0.0;
    int counter = 0;

    // Process each line of the file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string trash;          // Variable to store unused fields
        std::string temp_open;      // Variable to store the open price
        std::string temp_close;     // Variable to store the close price
        counter++;

        // Extract and discard date
        std::getline(ss, trash, ',');

        // Extract and store the open price
        std::getline(ss, temp_open, ',');
        std::cout << temp_open << std::endl;

        // Extract and discard high and low prices
        std::getline(ss, trash, ',');
        std::getline(ss, trash, ',');

        // Extract and store the close price
        std::getline(ss, temp_close, ',');
        std::cout << temp_close << std::endl;
        total_return_percentage += (std::stod(temp_close) - std::stod(temp_open)) / std::stod(temp_open);
    }
    
    // Close the file
    file.close();

    // Calculate the mean return
    double mean_return_percentage = total_return_percentage / static_cast<double>(counter);

    // Update the Asset object with the accumulated values
    if (asset_ptr)
        asset_ptr->set_mean_return(mean_return_percentage);

    return 0;
}