from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import numpy as np

A = np.array([[3.0825, 0, 0, 0], [0, 0.0405, 0, 0], [0, 0, 0.0271, -0.0031], \
    [0, 0, -0.0031, 0.0054]])
b = np.array([[2671], [135], [103], [19]])

# Réécriture des contraites sous une forme compréhensible par minimize :
linear_constraint = LinearConstraint([[-0.0401, -0.0162, -0.0039, 0.0002], \
    [-0.1326, -0.0004, -0.0034, 0.0006], [1.5413, 0, 0, 0], \
    [0, 0.0203, 0, 0], [0, 0, 0.0136, -0.0015], \
    [0, 0, -0.0016, 0.0027], [0.0160, 0.0004, 0.0005, 0.0002]], \
    [-np.inf]*7, [-92.6, 29.0, 2671, 135, 103, 19, 10])

def f(p):
    p = p.reshape((4, 1)) # précaution
    return float(0.5 * np.matmul(p.transpose(), np.matmul(A, p)) -\
        np.dot(b.transpose(), p))

x0 = np.ones((4,))
res = minimize(f, x0, method ='trust-constr', constraints=[linear_constraint], \
    options={'verbose': 1})
print(res.x)
res = [420.23259847 4025.00827192 2774.95770729 1393.98131029]
f_res = -1280286.6109881834