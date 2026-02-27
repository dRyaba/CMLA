"""
Неявная разностная схема для уравнения теплопроводности.

Решает задачу: u_t = a·u_xx + f(x, t) на [0, 1] × [0, 1]
с граничными и начальными условиями.
Использует метод прогонки (Thomas algorithm).
Аппроксимация: O(τ + h²).
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
A_COEFF = 0.019
TAU = 0.1
H = 0.1


def exact_solution(x: float, t: float) -> float:
    """Точное решение u(x, t)."""
    return -(x**4) + x + t * x + t**2 - t * np.exp(x)


def source_term(x: float, t: float) -> float:
    """Правая часть f(x, t) уравнения u_t = a·u_xx + f."""
    return (
        2 * t + x - np.exp(x)
        - A_COEFF * (-12 * x**2 - t * np.exp(x))
    )


def thomas_algorithm_step(
    t: float, x: np.ndarray, u_prev: np.ndarray, a_coef: float, b_coef: float, c_coef: float
) -> np.ndarray:
    """
    Один шаг неявной схемы (метод прогонки).

    Args:
        t: текущий момент времени
        x: сетка по пространству
        u_prev: решение на предыдущем временном слое
        a_coef, b_coef, c_coef: коэффициенты прогонки

    Returns:
        Решение на новом временном слое
    """
    n = len(x)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    u_new = np.zeros(n)

    u_new[0] = beta[0] = exact_solution(0, t)
    u_new[-1] = exact_solution(x[-1], t)

    for k in range(1, n - 1):
        d_k = (
            u_prev[k]
            + TAU * source_term(x[k], t)
            + (TAU * A_COEFF / (2 * H**2)) * (u_prev[k + 1] - 2 * u_prev[k] + u_prev[k - 1])
        )
        beta[k] = (c_coef * beta[k - 1] + d_k) / (b_coef - c_coef * alpha[k - 1])
        alpha[k] = a_coef / (b_coef - c_coef * alpha[k - 1])

    for k in range(n - 2, 0, -1):
        u_new[k] = alpha[k] * u_new[k + 1] + beta[k]

    return u_new


def main() -> None:
    """Основная функция: решение и визуализация."""
    t_grid = np.arange(0, 1, TAU)
    x_grid = np.arange(0, 1, H)

    a_coef = c_coef = (A_COEFF * TAU) / (2 * H**2)
    b_coef = 1 + (A_COEFF * TAU) / H**2

    u_prev = np.zeros(len(x_grid) + 1)
    for i in range(len(x_grid) + 1):
        u_prev[i] = exact_solution(i * H, 0)

    u_approx = np.zeros((len(x_grid), len(t_grid)))
    u_approx[:, 0] = u_prev[:len(x_grid)]

    max_err = 0.0
    for n in range(1, len(t_grid)):
        u_new = thomas_algorithm_step(t_grid[n], x_grid, u_prev, a_coef, b_coef, c_coef)
        u_approx[:, n] = u_new
        u_prev = deepcopy(u_new)

        for j in range(len(x_grid)):
            exact = exact_solution(x_grid[j], t_grid[n])
            max_err = max(max_err, abs(u_new[j] - exact))

    print(f"Максимальная погрешность: {max_err}")
    print(f"Погрешность аппроксимации: O(tau + h^2) = O({TAU + H**2})")

    x_mesh, t_mesh = np.meshgrid(t_grid, x_grid)
    exact_sol = np.array([
        [exact_solution(x_grid[j], t_grid[n]) for n in range(len(t_grid))]
        for j in range(len(x_grid))
    ])

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    fig1.canvas.manager.set_window_title("Exact solution")
    ax1.set_ylabel("x")
    ax1.set_xlabel("t")
    ax1.plot_surface(x_mesh, t_mesh, exact_sol)

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    fig2.canvas.manager.set_window_title("Approximate solution")
    ax2.set_ylabel("x")
    ax2.set_xlabel("t")
    ax2.plot_surface(x_mesh, t_mesh, u_approx)

    plt.show()


if __name__ == "__main__":
    main()
