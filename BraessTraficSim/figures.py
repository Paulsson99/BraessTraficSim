import matplotlib.pyplot as plt
import numpy as np


def SD(p_hat: np.ndarray, N: int) -> np.ndarray:
    """
    Calculate the standard diviation for the mean of a sample taken N times
    """
    return np.sqrt(p_hat * (1 - p_hat) / N)


network = [121, 131, 141, 151, 161]
prob_one_connection = np.array([0.392, 0.186, 0.101, 0.055, 0.029])
prob_fully_connected = np.array([0.392, 0.239, 0.134, 0.086, 0.030])

plt.errorbar(network, prob_one_connection, yerr=1.96*SD(prob_one_connection, 1000), capsize=5, capthick=2, label="One connection")
plt.errorbar(network, prob_fully_connected, yerr=1.96*SD(prob_fully_connected, 1000), capsize=5, capthick=2, label="Fully connected")

plt.legend()
plt.xticks(network, rotation=45)
plt.xlabel("Network configuration", size=14)
plt.ylabel("Fraction of simulation with the paradox", size=14)
plt.tight_layout()
plt.show()