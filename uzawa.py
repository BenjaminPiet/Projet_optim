#Mise en place des fonctions et valeurs numériques
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

def f(p):
    return float(0.5 * np.matmul(p.transpose(), np.matmul(A, p)) - np.dot(b.transpose(), p))

def c(p):
    return np.matmul(C.transpose(), p) - d

def grad_f(p):
    return np.matmul(A,p)+b
def grad_c(p):
    return C.transpose()

# Conditions de Wolfe
def wolfe_step(fun, grad_fun, xk, pk, c1 = 0.25, c2 = 0.75, M = 1000):
	l_moins = 0
	l_plus = 0
	f_xk = fun(xk)
	grad_f_xk = grad_fun(xk)
	li = 1 #0.0001
	i = 0
	while(i < M):
		if (fun(xk+li*pk)>(f_xk+c1*li*np.dot(grad_f_xk,pk))):
			l_plus = li
			li = (l_moins+l_plus)/2.0
		else:
			if (np.dot(grad_fun(xk+li*pk),pk) < c2*np.dot(grad_f_xk,pk)):
				l_moins = li
				if (l_plus == 0):
					li = 2*li
				else:
					li = (l_moins+l_plus)/2.0
			else:
				#print("Nb itérations : ", i)
				return li
		i = i + 1
	#print("Trop d'itérations de Wolfe")
	return li




def uzawa_wolfe_step(fun, grad_fun, c, grad_c, x0, rho, lambda0 = 1.0, max_iter = 100000, epsilon_grad_L = 1e-8):
	k = 0
	xk = x0
	lambdak = lambda0
	grad_Lagrangienk_xk = grad_fun(xk) + lambdak*grad_c(xk)
	while ((k<max_iter) and (np.linalg.norm(grad_Lagrangienk_xk)>epsilon_grad_L)):
		Lagrangienk = lambda x : fun(x) + lambdak*c(x)
		grad_Lagrangienk = lambda x : grad_fun(x) + lambdak*grad_c(x)
		grad_Lagrangienk_xk = grad_Lagrangienk(xk)
		pk = -grad_Lagrangienk_xk
		lk = wolfe_step(Lagrangienk, grad_Lagrangienk, xk, pk)
		xk = xk + lk*pk;        
		lambdak = np.maximum(0, lambdak + rho*c(xk))
		k = k + 1
	print("Nombre d'iterations : ", k)
	print("lambdak : ", lambdak)
	return xk

x0=np.array([0,0,0,0])
print(uzawa_wolfe_step(f,grad_f,c,grad_c,x0,0.1)
