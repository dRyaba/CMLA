import numpy as np
from math import pi, sin, cos
from numpy import fmod
from scipy.sparse import diags
from scipy.sparse.linalg import inv


def u(x, y):
    return x ** 2 + (np.cos(x * y)) ** 2


def f(x, y):
    return -2 * a + 2 * (a * (y ** 2) + b * (x ** 2)) * np.cos(2 * x * y)


a = 0.9
b = 1.1
h = 0.1
eps = 1e-3

N = int(1 / h)
size = N - 1

func_val = [f(i // size, fmod(i, size)) for i in range(size ** 2)]
func_val = np.array(func_val)
func_val1 = func_val
for i in range(size):
    func_val[i] += u(0, i)
    func_val[i * size] = u(i, 0)
    func_val[size*(size-1) + i] = u(N, i)
    func_val[i * (size-1)] = u(i, N)

a = a / h ** 2
b = b / h ** 2


def find_eigenvalue(r, eigen, matrix):
    A_r = matrix.dot(r)
    while np.linalg.norm(A_r - np.dot(eigen, r), np.inf) > eps:
        r = A_r / np.linalg.norm(A_r)
        A_r = matrix.dot(r)
        eigen = (np.dot(r, A_r) / np.dot(r, r))
    return eigen


def seidel():
    x = np.zeros(size ** 2)  # zero vector

    x_new = (inv(LD)).dot(func_val - U.dot(x))
    while np.linalg.norm(x_new - x) > eps:
        x = np.copy(x_new)
        # for i in range(size):
        #     s1 = sum(A[i][j] * x_new[j] for j in range(i))
        #     s2 = sum(A[i][j] * x[j] for j in range(i + 1, size))
        #     x_new[i] = (f(i // size, fmod(i, size)) - s1 - s2) / A[i][i]
        # x = x_new
        x_new = (inv(LD)).dot(func_val - U.dot(x))
    return x_new


minimal_actual = 4 * (a + b) * (sin(pi * h / 2)) * sin(pi * h / 2)
maximal_actual = 4 * (a + b) * (cos(pi * h / 2)) * cos(pi * h / 2)
print('actual_maximum = ', maximal_actual)
print('actual_minimum = ', minimal_actual)

D_main = [2 * (a + b)] * (size ** 2)
D_b = [-b] * (size ** 2 - 1)
for i in range(size, size ** 2 - 1, size):
    # if (fmod(i, size)) == 0:
    D_b[i - 1] = 0
D_a = [-a] * (size * (size - 1))
diagonals = [D_a, D_b, D_main, D_b, D_a]
A = diags(diagonals, [-size, -1, 0, 1, size])
LD_diagonals = [D_a, D_b, D_main]
LD = diags(LD_diagonals, [-size, -1, 0])
U_diagonals = [D_b, D_a]
U = diags(U_diagonals, [1, size])

r_0 = np.array([1] * (size ** 2))
max_eigen = (np.dot(r_0, A.dot(r_0)) / np.dot(r_0, r_0))
max_eigen = find_eigenvalue(r_0, max_eigen, A)
print("|max_eigen| = ", max_eigen)

first_norm = 4 * (a + b)
E = [1] * (size ** 2)

matr = 4 * (a + b) * diags(E) - A
min_eigen = (np.dot(r_0, matr.dot(r_0)) / np.dot(r_0, r_0))
min_eigen = 4 * (a + b) - find_eigenvalue(r_0, min_eigen, matr)
print("|min_eigen| = ", min_eigen)

res = seidel()
print("Seidel method:", res)
