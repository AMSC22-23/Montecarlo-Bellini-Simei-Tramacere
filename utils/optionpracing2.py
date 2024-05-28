import numpy as np

# Define asset price parameters
S0 = 91.2 # initial asset price
mu = 11.1141 # expected return
sigma = 0.2 # volatility
r = 0.05  # Tasso di interesse senza rischio
T = 1/12  # Tempo fino alla scadenza dell'opzione

# Define simulation parameters
n_simulation = 100000 # number of simulations
n_steps = 50 # number of steps in each simulation
dt = 1/n_steps # time step

# Generate random numbers
rn = np.random.normal(0, 1, [n_simulation, n_steps])

# Simulate asset prices
S = S0*np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rn)
S[:,0] = S0

# Calculate option payoff
K = 100 # strike price
payoff = np.maximum(S[:,-1] - K, 0)

# Calculate option price
C = np.exp(-r*T)*np.mean(payoff)

print(f"GOOGLE stock price: ", np.mean(S[:,-1]))
print(f"GOOGLE Option price: {C}")