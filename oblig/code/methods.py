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
        b = rng.random((rad))
    else: 
        W = np.random.randn(kol, 1)
        b = np.random.randn(rad)
    return W, b

def gradients():
    
    return dCdW, dCdb