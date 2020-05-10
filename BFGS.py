import numpy as np

A = np.array([[3.0825, 0, 0, 0], [0, 0.0405, 0, 0], [0, 0, 0.0271, -0.0031], \
	[0, 0, -0.0031, 0.0054]])
b = np.array([[2671], [135], [103], [19]])
C = np.array([[-0.0401, -0.0162, -0.0039, 0.0002], \
	[-0.1326, -0.0004, -0.0034, 0.0006], [1.5413, 0, 0, 0], \
	[0, 0.0203, 0, 0], [0, 0, 0.0136, -0.0015], \
	[0, 0, -0.0016, 0.0027], [0.0160, 0.0004, 0.0005, 0.0002]]).transpose()
d = np.array([[-92.6], [-29.0], [2671], [135], [103], [19], [10]])

def f(p):
	return float(0.5 * np.matmul(p.transpose(), np.matmul(A, p)) - np.dot(b.transpose(), p))

def c(p):
	return np.matmul(C.transpose(), p) - d

def grad_f(p):
	return np.matmul(A, p) - b

def grad_c(p):
	return C

def wolfe_step(fun, grad_fun, xk, pk, c1 = 0.25, c2 = 0.75, M = 1000):
	l_moins = 0
	l_plus = 0
	f_xk = fun(xk)
	grad_f_xk = grad_fun(xk)
	li = 0.01
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

def newton_BFGS(fun, grad_fun, c, grad_c, x0, lambda0, max_iter = 100000, epsilon_grad_L = 1e-3):
	k = 0
	xk = x0
	lambdak = lambda0
	Hk = np.identity(len(x0))
	grad_Lagrangienk_xk = grad_fun(xk) + np.matmul(grad_c(xk), lambdak)
	while ((k < max_iter) and (np.linalg.norm(grad_Lagrangienk_xk) > epsilon_grad_L)):
		Lagrangienk = lambda x : fun(x) + np.dot(lambdak.T, c(x))
		grad_Lagrangienk = lambda x : grad_fun(x) + np.matmul(grad_c(x), lambdak)
		grad_Lagrangienk_xk = grad_Lagrangienk(xk)
		pk = -np.matmul(Hk, grad_Lagrangienk_xk)
		lk = wolfe_step(Lagrangienk, grad_Lagrangienk, xk, pk)
		xk1 = xk + lk*pk
		grad_Lagrangienk_xk1 = grad_Lagrangienk(xk1)
		sk = xk1 - xk
		yk = grad_Lagrangienk_xk1 - grad_Lagrangienk_xk
		gammak = float(1.0/np.dot(yk.T, sk))
		Ak = np.identity(len(x0)) - gammak*np.multiply(sk[:, np.newaxis], yk)
		Bk = np.identity(len(x0)) - gammak*np.multiply(yk[:, np.newaxis], sk)
		Hk = np.matmul(np.dot(Ak, Hk), Bk) + gammak*np.multiply(sk[:, np.newaxis], sk)
		xk = xk1
		rhok1 = np.dot(grad_c(xk1), np.matmul(Hk, grad_c(xk1)))
		lambdak = np.maximum(0, lambdak + (1/rhok1)*c(xk1))
		k = k + 1
	print("Nombre d'iterations : ", k)
	print("lambda_k : ", lambdak)
	return xk

x0 = np.zeros((4, 1))
lambda0 = np.ones((7, 1))
print("Newton BFGS...")
x_newton_BFGS = newton_BFGS(f, grad_f, c, grad_c, x0, lambda0)
print("x_newton_BFGS : ", x_newton_BFGS)
print("c(x_newton_BFGS) : ", c(x_newton_BFGS))