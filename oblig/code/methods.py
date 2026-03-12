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
        Vekter W (matrise) og biaser b (skalar)
    """
    if seed is not None: 
        rng = np.random.default_rng(seed)
        W = rng.random((kol, 1))
        b = rng.random()
    else: 
        W = np.random.randn(kol, 1)
        b = np.random.randn()
    return W, b

def output(X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    """Ganger input med vekter og legger til bias. Kjører gjennom 
    sigmoidfunksjonen

    Args:
        X (np.ndarray): Input X. Kan være treningsdata. Shape: (nxp)
        W (np.ndarray): Vekter. Shape: (px1)
        b (float): Bias (skalar)

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
    dCdb = np.sum(delta, axis=0)
    return dCdW, dCdb

def gradient_step(W: np.ndarray, 
                 dCdW: np.ndarray, 
                 b: np.ndarray, 
                 dCdb: np.ndarray, 
                 eta: float = 0.01):
    """Gjør et steg med gradient descent

    Args:
        W (np.ndarray): W før steg
        dCdW (np.ndarray): gradient W
        b (np.ndarray): b før steg
        dCdb (np.ndarray): gradient b
        eta (float, optional): Steglengde. Defaults to 0.01.

    Returns:
        W: W etter steg
        b: b etter steg
    """
    W = W - eta * dCdW
    b = b - eta * dCdb
    return W, b

def fit_model(X: np.ndarray, 
              y: np.ndarray, 
              seed: int = None, 
              eta: float = 0.01, 
              n_iters: int=1000):
    """Funksjon for å tilpasse modell med gradient descent. Data blir transformert
    til riktig type og shape. Printer for hver 10ende prosent i treningsløkka. 

    Args:
        X (np.ndarray): Designmatrise
        y (np.ndarray): Respons
        seed (int, optional): Seed. Defaults to None.
        eta (float, optional): Læringsrate. Defaults to 0.01.
        n_iters (int, optional): Antall iterasjoner. Defaults to 1000.

    Returns:
        W: Vektor med vekter
        b: bias (skalar)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n, p = np.shape(X)
    W, b = vekter_bias(n, p, seed)
    
    step = n_iters // 10 
    
    for i in range(n_iters):
        y_hat = output(X, W, b)
        dW, db = gradients(y, y_hat, X)
        W, b = gradient_step(W, dW, b, db, eta)
        if (i+1) % step == 0:
            print(f"Fullført {i+1} steg av {n_iters}")
    return W, b