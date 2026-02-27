"""
Метод Зейделя для уравнения Пуассона.

Решает задачу: a·u_xx + b·u_yy = f(x, y) на единичном квадрате [0, 1]²
с граничными условиями Дирихле.
Использует разложение A = L + D + U и итерации (L+D)^{-1}(b - U·x).
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt

# Параметры задачи
A_COEFF = 0.9   # коэффициент при u_xx
B_COEFF = 1.1   # коэффициент при u_yy
H = 0.01        # шаг сетки
EPS = 1e-4      # точность итераций


def exact_solution(x: float, y: float) -> float:
    """Точное решение u(x, y)."""
    return x**2 + np.cos(x * y) ** 2


def source_term(x: float, y: float) -> float:
    """Правая часть f(x, y) уравнения a·u_xx + b·u_yy = f."""
    return (
        -2 * A_COEFF
        + 2 * (A_COEFF * y**2 + B_COEFF * x**2) * np.cos(2 * x * y)
    )


def build_rhs_vector(
    x_grid: np.ndarray, size: int, a_scaled: float, b_scaled: float
) -> np.ndarray:
    """
    Построение вектора правой части с учётом граничных условий.

    Args:
        x_grid: одномерная сетка
        size: число внутренних точек по одной оси
        a_scaled, b_scaled: коэффициенты, масштабированные на h²

    Returns:
        Вектор правой части длины size²
    """
    rhs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            rhs[i, j] = source_term(x_grid[i + 1], x_grid[j + 1])

    for i in range(size):
        rhs[0, i] += a_scaled * exact_solution(0, x_grid[i + 1])
        rhs[-1, i] += a_scaled * exact_solution(1, x_grid[i + 1])
        rhs[i, 0] += b_scaled * exact_solution(x_grid[i + 1], 0)
        rhs[i, -1] += b_scaled * exact_solution(x_grid[i + 1], 1)

    return rhs.ravel()


def gauss_seidel_solve(
    ld_inv: scipy.sparse.spmatrix,
    u_matrix: scipy.sparse.spmatrix,
    a_matrix: scipy.sparse.spmatrix,
    rhs: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int]:
    """
    Решение СЛАУ методом Зейделя: x_new = (L+D)^{-1}(b - U·x_old).

    Args:
        ld_inv: обратная к (L+D)
        u_matrix: верхняя треугольная часть U
        a_matrix: полная матрица (для проверки невязки)
        rhs: вектор правой части
        eps: требуемая точность по невязке

    Returns:
        (решение, число итераций)
    """
    x = np.zeros_like(rhs)
    x_new = ld_inv.dot(rhs - u_matrix.dot(x))
    n_iter = 1

    while np.linalg.norm(a_matrix.dot(x_new) - rhs, np.inf) > eps:
        x = np.copy(x_new)
        x_new = ld_inv.dot(rhs - u_matrix.dot(x))
        n_iter += 1

    return x_new, n_iter


def main() -> None:
    """Основная функция: решение и визуализация."""
    start_time = time.time()

    x_grid = np.arange(0, 1 + 1e-6, H)
    x_mesh, y_mesh = np.meshgrid(x_grid, x_grid)

    n_total = x_grid.size
    size = n_total - 2  # внутренние точки

    a_scaled = A_COEFF / H**2
    b_scaled = B_COEFF / H**2

    rhs = build_rhs_vector(x_grid, size, a_scaled, b_scaled)

    # Пятидиагональная матрица
    d_main = [2 * (a_scaled + b_scaled)] * (size**2)
    d_b = [-b_scaled] * (size**2 - 1)
    for i in range(size, size**2 - 1, size):
        d_b[i - 1] = 0
    d_a = [-a_scaled] * (size * (size - 1))

    diagonals = [d_a, d_b, d_main, d_b, d_a]
    a_matrix = diags(diagonals, [-size, -1, 0, 1, size])

    ld_diagonals = [d_a, d_b, d_main]
    ld_matrix = scipy.sparse.csc_matrix(diags(ld_diagonals, [-size, -1, 0]))

    u_diagonals = [d_b, d_a]
    u_matrix = diags(u_diagonals, [1, size])

    inv_start = time.time()
    ld_inv = inv(ld_matrix)
    print(f"Время обращения матрицы: {time.time() - inv_start:.4f} с")

    solution, n_iter = gauss_seidel_solve(
        ld_inv, u_matrix, a_matrix, rhs, EPS
    )
    print(f"Число итераций: {n_iter}")

    solution = np.reshape(solution, (size, size))

    # Сборка полного решения с границами
    approx_sol = np.zeros((n_total, n_total))
    exact_sol = np.zeros((n_total, n_total))

    for i in range(n_total):
        approx_sol[0, i] = exact_solution(0, x_grid[i])
        approx_sol[i, 0] = exact_solution(x_grid[i], 0)
        approx_sol[-1, i] = exact_solution(1, x_grid[i])
        approx_sol[i, -1] = exact_solution(x_grid[i], 1)
        for j in range(n_total):
            exact_sol[i, j] = exact_solution(x_grid[i], x_grid[j])

    for i in range(size):
        for j in range(size):
            approx_sol[i + 1, j + 1] = solution[i, j]

    elapsed = time.time() - start_time
    print(f"Общее время: {elapsed:.4f} с")

    max_err = np.max(np.abs(exact_sol - approx_sol))
    print(f"Максимальная погрешность: {max_err}")

    # Визуализация
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    fig1.canvas.manager.set_window_title("Exact solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.plot_surface(y_mesh, x_mesh, exact_sol, color="b")

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    fig2.canvas.manager.set_window_title("Approximate solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.plot_surface(y_mesh, x_mesh, approx_sol, color="r")

    plt.show()


if __name__ == "__main__":
    main()
