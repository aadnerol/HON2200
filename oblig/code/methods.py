import numpy as np

def sigmoid(z: float) -> float:
    """Sigmoid-funksjonen. Tar høyde for overflow med np.where()

    Args:
        z (float): Input

    Returns:
        float: Funksjonsverdi
    """
    return np.where (z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def vekter_bias(rad: int, kol: int, seed: int = None):
    """Lager vekter og biaser.

    Args:
        rad (int): Antall rader i designmatrisen
        kol (int): Antall kolonner i designmatrisen
        seed (int): Seed for reproduserbarhet. Optional

    Returns:
        Vekter W (matrise) og biaser b (vektor)
    """
    if seed: 
        rng = np.random.default_rng(seed)
        W = rng.random((kol, 1))
        b = rng.random((rad, 1))
    else: 
        W = np.random.randn(kol, 1)
        b = np.random.randn(rad, 1)
    return W, b

def output(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ganger input med vekter og legger til bias. Kjører gjennom 
    sigmoidfunksjonen

    Args:
        X (np.ndarray): Input X. Kan være treningsdata. Shape: (nxp)
        W (np.ndarray): Vekter. Shape: (px1)
        b (np.ndarray): Bias. Shape: (nx1)

    Returns:
        float: y_hat. Shape: (nx1)
    """
    z = X @ W + b
    y_hat = sigmoid(z)
    return y_hat

def gradients(y: np.ndarray, y_hat: np.ndarray, X: np.ndarray):
    """Function for computing gradients. 

    Args:
        y (np.ndarray): y. Shape nx1
        y_hat (np.ndarray): y_hat (output). Shape nx1
        X (np.ndarray): X. Shape nxp

    Returns:
        dCdW: gradient. Shape px1
        dCdb: gradient. Shape nx1
    """
    n = np.shape(y)[0]
    delta = - (y-y_hat) / n
    dCdW = X.T @ delta
    dCdb = np.sum(delta, axis = 1)
    return dCdW, dCdb