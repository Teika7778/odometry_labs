import numpy as np
import math
import matplotlib.pyplot as plt


# Функция невязки
def res(y, x, theta, n):
    x_par = (n.T @ x) * n
    x_ort = x - x_par
    x_cross = np.cross(n, x)
    model = x_par + np.cos(theta) * x_ort + x_cross * np.sin(theta)
    return y - model


# Якобиан
def jac(x, theta, n):
    x_par = (n.T @ x) * n
    x_ort = x - x_par
    x_cross = np.cross(n, x)
    return np.sin(theta) * x_ort - x_cross * np.cos(theta)


def gn(init, max_iter, eps_1, eps_2, X, Y, n):
    # Y как параметр функции является матрицей, но математически это вектор
    N = X.shape[0]
    history = []
    cur = init
    for _ in range(max_iter):
        history.append(cur)
        r = np.zeros((3 * N, 1))
        J = np.zeros((3 * N, 1))
        for i in range(N):
            x = X[i]
            y = Y[i]
            r_loc = res(y, x, cur, n)
            J_loc = jac(x, cur, n)
            for j in range(3):
                r[3 * i + j] = r_loc[j]
                J[3 * i + j] = J_loc[j]
        g = J.T @ r
        if np.linalg.norm(g) < eps_1:
            return history
        H = J.T @ J
        delta = - (g / H)[0, 0]
        if np.abs(delta) < eps_2:
            return history
        cur += delta
    return history


def generate_X(N, l, u):
    return np.random.uniform(l, u, size=(N, 3))


def generate_Y(X, theta, n, noise=0.1):
    N = X.shape[0]
    Y = np.zeros((N, 3))
    for i in range(N):
        x = X[i]
        x_par = (n.T @ x) * n
        x_ort = x - x_par
        x_cross = np.cross(n, x)
        model = x_par + np.cos(theta) * x_ort + x_cross * np.sin(theta) + np.random.normal(loc=0.0, scale=np.abs(
            X).max() * noise, size=3)
        Y[i] = model
    return Y


true = math.pi * 2.0 / 3.0
n = np.array([1.0, 0.5, 0.25])
n = n / np.linalg.norm(n)

X = generate_X(10000, 10.0, -10.0)
Y = generate_Y(X, true, n)

hist = gn(0.0, 100, 1e-8, 1e-8, X, Y, n)
histf = [float(v) for v in hist]
print(true - histf[-1])
print(histf)


def plot_res(h, X, y, n):
    ox = list(range(len(h)))
    oy = [0] * len(ox)
    N = X.shape[0]

    for t in range(len(ox)):
        r = np.zeros((3 * N, 1))
        for i in range(N):
            x = X[i]
            y = Y[i]
            r_loc = res(y, x, h[t], n)
            for j in range(3):
                r[3 * i + j] = r_loc[j]
        oy[t] = float(np.linalg.norm(r)) ** 2

    plt.plot(ox, oy)
    plt.title("Зависимость суммы квадратов невязок от номера итерации")
    plt.xlabel("k")
    plt.ylabel("S(k)")
    plt.grid(True)
    plt.show()


def plot_val(h):
    ox = list(range(len(h)))
    oy = h
    plt.plot(ox, oy)
    plt.title("Зависимость оценки параметра от номера итерации")
    plt.xlabel("k")
    plt.ylabel("$theta_k$")
    plt.grid(True)
    plt.show()


plot_res(histf, X, Y, n)

hist = gn(-1000.0, 100, 1e-8, 1e-8, X, Y, n)
histf = [float(v) for v in hist]

plot_res(histf, X, Y, n)
plot_val(histf)

histfm = [(num + math.pi) % math.tau - math.pi for num in histf]
print(histfm)
plot_val(histfm)

hist = gn(1000.0, 100, 1e-8, 1e-8, X, Y, n)
histf = [float(v) for v in hist]

plot_val(histf)

histfm = [(num + math.pi) % math.tau - math.pi for num in histf]
print(histfm)
plot_val(histfm)
