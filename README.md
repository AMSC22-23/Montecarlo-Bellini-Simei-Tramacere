# <div align="center"> MonteCarlo Integration </div>
## <div align="center"> Emanuele Bellini, Luca Simei, Luca Tramacere </div>

This is the project for the Advanced Methods for Scientific Computing course @ Politecnico di Milano. Objective of this project is to compute the approximation of an integral over a domain in n dimensions using a MonteCarlo Algorithm.

### To compile

```bash
make
rm ./runner
ln -s build/bin/runner runner
./runner
```

### What to expect
The user can choose the integration domain from the following options:
- [x] (hs) HyperSphere
- [x] (hr) HyperRectangle
- [x] (hc) HyperCube

He can then choose:

- The number of random points to generate
- The dimension in which the integration domain is located
- Some parameters related to the integration domain
- The function to be integrated

The program will provide the approximate value of the integral and the time required for the calculation.

### Notes
The higher the number of random points the user chooses to generate, the slower the program will be, but the result will be more accurate. The code is parallelized using OpenMP. The user can freely select the function they want to integrate, thanks to the [muParserX](https://github.com/beltoforion/muparserx) library.

### Next steps
- Enhance domain flexibility, like domain location and other types of domains
- Implement feature for managing complex numbers
- Application in a real world case
