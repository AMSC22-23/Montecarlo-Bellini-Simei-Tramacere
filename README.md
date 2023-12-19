# <div align="center"> MonteCarlo Integration </div>
## <div align="center"> Emanuele Bellini, Luca Simei, Luca Tramacere </div>

This is the project for the Advanced Methods for Scientific Computing course @ Politecnico di Milano. Objective of this project is to compute the approximation of an integral over a domain in n dimensions using a MonteCarlo Algorithm.

### To compile

```bash
g++ -std=c++17 main.cpp HyperSphere.cpp input_manager.cpp mc_integrator.cpp -I{your path to muparser folder}/muparser-2.3.4/include -o main
```
