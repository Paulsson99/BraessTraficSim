import matplotlib.pyplot as plt
import numpy as np


def SD(p_hat: np.ndarray, N: int) -> np.ndarray:
    """
    Calculate the standard diviation for the mean of a sample taken N times
    """
    return np.sqrt(p_hat * (1 - p_hat) / N)


wide_network = [121, 131, 141, 151, 161]
wide_prob_one_connection = np.array([0.392, 0.186, 0.101, 0.055, 0.029])
wide_prob_fully_connected = np.array([0.392, 0.239, 0.134, 0.086, 0.030])

wide_fig, wide_ax = plt.subplots(1)

wide_ax.errorbar(range(len(wide_network)), wide_prob_one_connection, yerr=1.96*SD(wide_prob_one_connection, 1000), capsize=5, capthick=2, label="One connection")
wide_ax.errorbar(range(len(wide_network)), wide_prob_fully_connected, yerr=1.96*SD(wide_prob_fully_connected, 1000), capsize=5, capthick=2, label="Fully connected")

wide_ax.legend()
wide_ax.set_xticks(range(len(wide_network)), rotation=45, labels=wide_network)
wide_ax.set_xlabel("Network configuration", size=14)
wide_ax.set_ylabel("Fraction of simulation with the paradox", size=14)
wide_fig.tight_layout()

long_network = [121, 1221, 12221, 122221, 1222221]
long_prob_one_connection = np.array([0.370, 0.133, 0.027, 0.006, 0.003])
long_prob_fully_connected = np.array([0.370, 0.171, 0.031, 0.011, 0.003])

long_fig, long_ax = plt.subplots(1)

long_ax.errorbar(range(len(long_network)), long_prob_one_connection, yerr=1.96*SD(long_prob_one_connection, 1000), capsize=5, capthick=2, label="One connection")
long_ax.errorbar(range(len(long_network)), long_prob_fully_connected, yerr=1.96*SD(long_prob_fully_connected, 1000), capsize=5, capthick=2, label="Fully connected")

long_ax.legend()
long_ax.set_xticks(range(len(long_network)), rotation=45, labels=long_network)
long_ax.set_xlabel("Network configuration", size=14)
long_ax.set_ylabel("Fraction of simulation with the paradox", size=14)
long_fig.tight_layout()

plt.show()