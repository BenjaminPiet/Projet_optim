import numpy as np

A = np.array([[3.0825, 0, 0, 0], [0, 0.0405, 0, 0], [0, 0, 0.0271, -0.0031], \
     [0, 0, -0.0031, 0.0054]])
b = np.array([[2671], [135], [103], [19]])
C = np.array([[-0.0401, -0.0162, -0.0039, 0.0002], \
    [-0.1326, -0.0004, -0.0034, 0.0006], [1.5413, 0, 0, 0], \
    [0, 0.0203, 0, 0], [0, 0, 0.0136, -0.0015], \
    [0, 0, -0.0016, 0.0027], [0.0160, 0.0004, 0.0005, 0.0002]]).transpose()
d = np.array([[-92.6], [-29.0], [2671], [135], [103], [19], [10]])
eps = 1e-8 # paramètre donnant la valeur maximale des composantes
# de la direction pk pour que pk ne soit pas considérée comme nulle.

def xk_is_solution(xk, Wk):
    '''
    Etape 1 de l'algorithme des contraintes actives.
    Détermine, grâce aux conditions KKT, si xk est une solution du problème
    d'optimisation restreint aux contraintes contenues dans Wk.
    '''
    CC = C[:, Wk]
    dd = d[Wk, :]
    rg = np.linalg.matrix_rank(CC)
    if rg == len(Wk):
        # Les contraintes actives sont qualifiées
        # Conditions de relâchement
        Wk_systeme = Wk.copy()
        for i, c_i in enumerate([CC[:, j] for j in range(len(Wk))]):
            if np.dot(c_i, xk) -  dd[i, 0] != 0:
                # Lambda_i = 0 : on retire la contrainte Wk_systeme[i]
                del Wk_systeme[Wk_systeme.index(Wk[i])]
        Lambda = np.zeros((len(Wk), 1))
        print(Wk_systeme)
        if len(Wk_systeme) != 0:
            CC_systeme = C[:, Wk_systeme]
            dd_systeme = d[Wk_systeme, :]
            Lambda_systeme = np.linalg.solve(CC_systeme.transpose(), dd_systeme)
            for j in Wk:
                if j in Wk_systeme:
                    Lambda[Wk.index(j), 0] = Lambda_systeme[Wk_systeme.index(j), 0]
        return all([Lambda[i, 0] > 0 for i in range(len(Wk))]), Lambda
    else:
        # Les contraintes actives ne sont pas qualifiées
        return False, None


W0 = [0, 1, 2, 3] # arbitraire mais permet de n'avoir qu'un point de départ possible
# car le système à résoudre contient 4 équations pour 4 inconnues.
# Recherche de p0 où ces 4 contraintes sont actives
CC = C[:, W0]
dd = d[W0, :]
x0 = np.linalg.solve(CC.transpose(), dd)

Wk = [0, 1, 3, 4]
xk = np.array([[0], [1], [10], [0]])