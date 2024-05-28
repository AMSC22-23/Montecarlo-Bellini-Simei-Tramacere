import numpy as np
from scipy.stats import norm

# Definizione dei parametri
S0 = np.array( [101.61, 278.34, 162.47, 5.35])  # Prezzi iniziali delle 4 stock
K = 0  # Prezzo di esercizio dell'opzione
r = 0.05  # Tasso di interesse senza rischio
sigma = np.array([0.0182483, 0.0211878, 0.0118315, 0.0393723])  # Volatilità delle 4 stock
rho = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])  # Matrice di correlazione
T = 3  # Tempo fino alla scadenza dell'opzione
n = 100000  # Numero di simulazioni Monte Carlo

# Simulazione Monte Carlo per generare i percorsi dei 4 sottostanti
np.random.seed(0)
z = np.random.multivariate_normal(np.zeros(4), rho, (n,))
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

# Calcolo del payoff dell'opzione call europea
payoff = np.maximum(np.mean(ST, axis=1) - K, 0)

# Calcolo del prezzo dell'opzione come il valore attuale medio dei payoff
C = np.exp(-r * T) * np.mean(payoff)

print(f"Il prezzo stimato dell'opzione call europea è: {C}")