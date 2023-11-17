#include <iostream>

int main() {
    String name;
    std::cout << "What's your name?" << std::endl;
    std::cin >> name;
    std::cout << "Hello" << name << "!" << std::endl;
    return 0;
}