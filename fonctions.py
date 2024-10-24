import numpy as np


def creeVec(taille):
    """
    -> Fonction pour créer un vecteur de taille donnée avec des valeurs aléatoires distribuées selon une loi normale standard.
    -> Input : taille (int) - La taille du vecteur à générer.
    -> Output: np.ndarray - Un vecteur aléatoire de taille 'taille'.
    """
    return np.random.randn(taille)

def manhatten(v1,v2):
    """
    -> Fonction pour calculer la distance de Manhattan entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance de Manhattan entre les deux vecteurs.
    """
    return np.sum(np.abs(v1-v2))

def euclidienne(v1,v2):
    """
    -> Fonction pour calculer la distance euclidienne entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance euclidienne entre les deux vecteurs.
    """
    return np.sqrt(np.sum((v1-v2)**2))

def tchebychev(v1,v2):

    """
    -> Fonction pour calculer la distance de tchebychev entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance de tchebychev entre les deux vecteurs.
    """
    return np.max(np.abs(v1-v2))

def intersectionX(v1,v2):
    """
    -> Fonction pour calculer l'intersection  par rapport à v1 entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - L'intersection  par rapport à v1 entre les deux vecteurs.
    """
    return np.sum(np.minimum(v1,v2))/np.sum(v1)

def intersectionY(v1,v2):
    """
    -> Fonction pour calculer l'intersection  par rapport à v2 entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - L'intersection  par rapport à v2 entre les deux vecteurs.
    """
    return np.sum(np.minimum(v1,v2))/np.sum(v2)
def khi2(v1,v2):
    """
    -> Fonction pour calculer la distance du Khi² entre deux vecteurs.
    -> Input : v1, v2 (np.ndarray) - Les deux vecteurs à comparer.
    -> Output: float - La distance Khi² entre les deux vecteurs.
    """
    return np.sum((v1-v2)**2/(v1+v2)**2)
def minkowski(v1,v2,p=1.5):
    return (np.sum(np.abs(v1-v2)**p))**(1/p)

def affichage(v1,v2,p=1.5):
    
    print("pour les vecteurs v1 :",v1 ,"et v2 : ",v2)
    print("manhatten   :" ,manhatten(v1,v2))
    print("euclidienne :" ,euclidienne(v1,v2))
    print("tchebychev  :" ,tchebychev(v1,v2))
    print("intersection:" ,intersectionX(v1,v2))
    print("khi2        :" ,intersectionY(v1,v2))
    print("minkowski   :" ,minkowski(v1,v2,p))

def affichage_formate(v1,v2,p=1.5):
    print("-------------------------------------------------------------------------------------------------------------------")
    print("|  Distance  |manhatten |euclidienne |tchebychev|intersectionX|intersectionY| khi2 |minkowski|")
    if np.array_equal(v1,v2) :
        print(f"|v identiques|  {manhatten(v1,v2):.4f}  |    {euclidienne(v1,v2):.4f}  |   {tchebychev(v1,v2):.4f} |    {intersectionX(v1,v2):.4f}   |   {intersectionY(v1,v2):.4f}    |{khi2(v1,v2):.4f}|   {minkowski(v1,v2,p):.4f}|")
    else:
        print(f"|v differents|  {manhatten(v1,v2):.4f}  |    {euclidienne(v1,v2):.4f}  |   {tchebychev(v1,v2):.4f} |    {intersectionX(v1,v2):.4f}   |   {intersectionY(v1,v2):.4f}    |{khi2(v1,v2):.4f}|   {minkowski(v1,v2,p):.4f}|")
v1 = np.array([2.7,4.3,0.2,9,-4])
v2 = np.array([7.6,5.8,-3.2,9.7,12.3])

affichage_formate(v1,v2,1.5)

affichage_formate(v1,v1)
print(intersectionX)