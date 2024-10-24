import numpy as np


def creeVec(taille):
    """
    -> Fonction pour créer un vecteur de taille donnée avec des valeurs aléatoires distribuées selon une loi normale standard.
    -> Input : taille (int) - La taille du vecteur à générer.
    -> Output: np.ndarray - Un vecteur aléatoire de taille 'taille'.
    """
    return np.random.randn(taille)

def manhattan(v1,v2):
    """
    -> Fonction pour calculer la distance de Manhattan entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance de Manhattan entre les deux vecteurs.
    """
    print('manhat')
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sum(np.abs(v1-v2))

def euclidienne(v1,v2):
    """
    -> Fonction pour calculer la distance euclidienne entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance euclidienne entre les deux vecteurs.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.sum((v1-v2)**2))

def tchebychev(v1,v2):

    """
    -> Fonction pour calculer la distance de tchebychev entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance de tchebychev entre les deux vecteurs.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.max(np.abs(v1-v2))

def intersectionX(v1,v2):
    """
    -> Fonction pour calculer l'intersection  par rapport à v1 entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - L'intersection  par rapport à v1 entre les deux vecteurs.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sum(np.minimum(v1,v2))/np.sum(v1)

def intersectionY(v1,v2):
    """
    -> Fonction pour calculer l'intersection  par rapport à v2 entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - L'intersection  par rapport à v2 entre les deux vecteurs.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sum(np.minimum(v1,v2))/np.sum(v2)
def khi2(v1,v2):
    """
    -> Fonction pour calculer la distance du Khi² entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance Khi² entre les deux vecteurs.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    epsilon = 1e-10
    return np.sum((v1-v2)**2/((v1+v2)+epsilon)**2)
def minkowski(v1,v2,p=1.5):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return (np.sum(np.abs(v1-v2)**p))**(1/p)