#Mise en place des fonctions et valeurs numériques
import numpy as np

A = np.array([[3.0825, 0, 0, 0], [0, 0.0405, 0, 0], [0, 0, 0.0271, -0.0031], \
	[0, 0, -0.0031, 0.0054]])
b = np.array([[2671], [135], [103], [19]])
C = np.array([[-0.0401, -0.1326, 1.5413, 0.0, 0.0, 0.0, 0.0160], \
	[-0.0162, -0.0004, 0.0, 0.0203, 0.0, 0.0, 0.0004], \
	[-0.0039, -0.0034, 0.0, 0.0, 0.0136, -0.0016, 0.0005], \
	[0.0002, 0.0006, 0.0, 0.0, -0.0015, 0.0027, 0.0002]])
d = np.array([[-92.6], [-29.0], [2671], [135], [103], [19], [10]])
eps = 1e-8 # paramètre donnant la valeur maximale des composantes
# de la direction pk pour que pk ne soit pas considérée comme nulle.

def f(p):
	return float(0.5 * np.matmul(p.T, np.matmul(A, p)) - np.dot(b.T, p))

def c(p):
	return np.matmul(C.T, p) - d

def grad_f(p):
	return np.matmul(A, p) - b

def grad_c(p):
	return C

# Conditions de Wolfe
def wolfe_step(fun, grad_fun, xk, pk, c1 = 0.25, c2 = 0.75, M = 1000):
	l_moins = 0
	l_plus = 0
	f_xk = fun(xk)
	grad_f_xk = grad_fun(xk)
	li = 0.001
	i = 0
	while(i < M):
		if (fun(xk + li*pk) > (f_xk + c1*li*np.dot(grad_f_xk.T, pk))):
			l_plus = li
			li = (l_moins + l_plus)/2.0
		else:
			if (np.dot(grad_fun(xk + li*pk).T, pk) < c2*np.dot(grad_f_xk.T, pk)):
				l_moins = li
				if (l_plus == 0):
					li = 2*li
				else:
					li = (l_moins + l_plus)/2.0
			else:
				return li
		i = i + 1
	return li


def uzawa_wolfe_step(fun, grad_fun, c, grad_c, x0, rho, lambda0, max_iter = 5000, epsilon_grad_L = 1e-3):
	k = 0
	xk = x0
	lambdak = lambda0
	grad_Lagrangienk_xk = grad_fun(xk) + np.matmul(grad_c(xk), lambdak)
	while ((k < max_iter) and (np.linalg.norm(grad_Lagrangienk_xk) > epsilon_grad_L)):
		print(np.linalg.norm(grad_Lagrangienk_xk))
		Lagrangienk = lambda x : fun(x) + np.dot(lambdak.T, c(x))
		grad_Lagrangienk = lambda x : grad_fun(x) + np.matmul(grad_c(x), lambdak)
		grad_Lagrangienk_xk = grad_Lagrangienk(xk)
		pk = -grad_Lagrangienk_xk
		lk = wolfe_step(Lagrangienk, grad_Lagrangienk, xk, pk)
		xk = xk + lk*pk
		lambdak = np.array([[max(0, lambdak[i, 0] + rho*c(xk)[i, 0])] for i in range(7)]) # projection sur R+^7
		k = k + 1
	print("Nombre d'iterations : ", k)
	print("lambdak : ", lambdak)
	return xk

x0 = np.ones((4, 1))
lambda0 = 100*np.ones((7, 1))
print(uzawa_wolfe_step(f, grad_f, c, grad_c, x0, 0.1, lambda0))


def uzawa_fixed_step(fun, grad_fun, c, grad_c, x0, l, rho, lambda0, max_iter = 100000, epsilon_grad_L = 1e-8):
	k = 0
	xk = x0
	lambdak = lambda0
	grad_Lagrangien_xk = grad_fun(xk) + np.matmul(grad_c(xk), lambdak)
	while ((k < max_iter) and (np.linalg.norm(grad_Lagrangien_xk) > epsilon_grad_L)):
		print(np.linalg.norm(grad_Lagrangien_xk))
		grad_Lagrangien_xk = grad_f(xk) + np.dot(lambdak.T, c(xk))
		pk = -grad_Lagrangien_xk
		xk = xk + l*pk
		lambdak = np.array([[max(0, lambdak[i, 0] + rho*c(xk)[i, 0])] for i in range(7)]) # projection sur R+^7
		k = k + 1
	print("Nombre d'iterations : ", k)
	print("lambdak : ", lambdak)
	return xk

#print("Uzawa fixed step...")
#x_fixed_step = uzawa_fixed_step(f, grad_f, c, grad_c, x0, 0.001, 0.1, lambda0)
#print("x_fixed_step : ", x_fixed_step)
#print("c(x_fixed_step) : ", c(x_fixed_step))


