#include <iostream>
#include <string>

int main() {
    std::string nome;
    std::cout << "What's your name?" << std::endl;
    std::cin >> nome;
    std::cout << "Hello " << nome << "!" << std::endl;
    return 0;
}