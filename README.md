# <div align="center"> MonteCarlo Integration </div>
## <div align="center"> Emanuele Bellini, Luca Simei, Luca Tramacere </div>

This is the project for the Advanced Methods for Scientific Computing course @ Politecnico di Milano. Objective of this project is to compute the approximation of an integral over a domain in n dimensions using the MonteCarlo Integration.

### Monte Carlo Integration for Option Pricing

Monte Carlo methods encompass a broad class of computational techniques that utilize random sampling to derive numerical solutions. These methods are particularly valuable for solving a wide range of problems that might be analytically intractable or computationally intensive using tradi-tional deterministic approaches.

Building on the general principles of Monte Carlo methods, Monte Carlo Integration is a specific application designed to estimate the integral of a function over a chosen domain using random sampling. This technique is particularly useful for high-dimensional and complex domains where traditional numerical integration methods become impractical.

Monte Carlo methods play an essential role in option pricing, particularly through Monte Carlo Integration. Option pricing, or the valuation of options, constitutes a fundamental pillar in financial practice. It enables the determination of the intrinsic value of a contract that grants its holder the right, but not the obligation, to buy or sell a financial asset (the underlying) at a predetermined price (the
strike price) by a specific date (the option’s expiration).

By generating a large number of random sample paths for the underlying asset’s price, Monte Carlo methods can model the potential outcomes and calculate the expected payoff of the option. This stochastic approach provides a flexible and powerful tool for tackling the complexities inherent in option pricing, making it indispensable in modern financial practice.

### Compile and Run

1. Create a build directory
```bash
mkdir build
cd build
```

2. Generate the build files using cmake
```bash
cmake ..
```

3. Compile the project
```bash
make
```

4a. Run the executable with OpenMP
```bash
./mainOmp
```

4b. If an NVIDIA GPU is found, run the executable with CUDA
```
./mainCUDA
```

### Purpose of the Project
The project aims to showcase the versatility and practical applications of Monte Carlo Integration through two distinct approaches: a finance-oriented project and a general implementation. 

In the finance-oriented project, Monte Carlo Integration serves as a powerful tool for pricing finan-cial options, particularly those dependent on multiple underlying assets. Leveraging the concept of Brownian motion and geometric Brownian motion models, the project simulates asset price paths over time to estimate option payoffs. By computing integrals of hyperrectangles in dimensions corresponding to the number of assets involved, the project facilitates the valuation of complex
multi-asset options. These computations are complemented by data extracted from CSV files or provided by the user, offering flexibility and real-world applicability. Moreover, the project explores parallelization techniques to optimize the computation process, ensuring efficient and accurate re- sults even for large volumes of data.

In contrast, the general implementation, presented at the hands-on, focuses on performing Monte Carlo Integration over selected domains, such as the hypercube, hypersphere, and hyperrectangle. This approach offers a comprehensive exploration of Monte Carlo Integration’s capabilities across different geometric shapes, allowing for the approximate calculation of integrals in C++ over a variable number of dimensions.

### Contributors
[Emanuele Bellini](https://github.com/EmanueleBellini99)

[Luca Simei](https://github.com/luca-simei)

[Luca Tramacere](https://github.com/trama00)

### Credits
[muParserX](https://github.com/beltoforion/muparserx)
