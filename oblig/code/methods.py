import numpy as np

def sigmoid(z: float) -> float:
    """Sigmoid-funksjonen

    Args:
        z (float): Input

    Returns:
        float: Funksjonsverdi
    """
    return np.exp(z) / (1 + np.exp(z))