import numpy as np
import matplotlib.pyplot as plt


def reflection(x):
    n = x.shape[0]
    sign = 1 if x[0] >= 0 else -1

    alpha = -1 * sign * np.linalg.norm(x)
    v = x.copy()
    v[0] -= alpha

    if np.linalg.norm(v) == 0:
        return np.eye(n)
    u = v / np.linalg.norm(v)
    H = np.eye(n) - 2 * np.outer(u, u)

    return H


def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(min(m, n)):
        x = R[i:, i]
        H_i = reflection(x)

        H = np.eye(m)
        H[i:, i:] = H_i

        R = H @ R
        Q = Q @ H

    return Q, R


def lstsq_QR(J, r):
    m, n = J.shape
    Q, R = householder_qr(J)  # Q: m x m, R: m x n
    # print(R)
    Q1 = Q[:, :n]  # Q1: m x n
    R1 = R[:n, :]  # R1: n x n

    # J = QR = Q1R1 (IDK)

    rhs = Q1.T @ r
    Delta = np.zeros(n)

    for i in range(n - 1, -1, -1):
        Delta[i] = (rhs[i] - R1[i, i + 1:n] @ Delta[i + 1:n]) / R1[i, i]

    return Delta


def lstsq_SVD(J, r):
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    # U: m x k | k = min(m,n) = n (уравнений должно быть больше)
    # S: k x k
    # c: k x n
    s_inv = np.zeros_like(s)
    for i in range(s.shape[0]):
        if s[i] > 1e-8:
            s_inv[i] = 1.0 / s[i]

    return Vt.T @ (s_inv * (U.T @ r))


def lstsq_inv(J, r):
    return np.linalg.inv(J.T @ J) @ J.T @ r


def compare(J, r):
    np.set_printoptions(precision=10, suppress=False)
    print("=" * 15)
    print("J =", J)
    print("r =", r)
    print("Cond(J) =", np.linalg.cond(J))
    print()
    funcs = {
        "Обычная формула": lstsq_inv,
        "QR разложение": lstsq_QR,
        "SVD разложение": lstsq_SVD
    }

    for f in funcs:
        print(f"{f}:")
        try:
            res = funcs[f](J, r)

            if not np.all(np.isfinite(res)):
                print("Failed")
                print()
                continue

            print(res)
            print("Error:", np.linalg.norm(J.T @ J @ res - J.T @ r))

        except Exception:
            print("Failed")
        print()
    print("=" * 15)
    print()


def gauss_newton(X, y, f, df, x0, step, iters=1):
    """
    В рамках данной работы будет сделана только одна итерация ГН.
    """
    current = x0
    for _ in range(iters):
        m, n = X.shape
        r = np.zeros(m)
        J = np.zeros((m, x0.shape[0]))
        for i in range(m):
            r[i] = f(X[i], current) - y[i]
            J[i] = df(X[i], current)

        delta = step(J, r)
        current = current - delta
    return current


# Стандартная матрица
J = np.array([
    [1, 2],
    [2, 1],
    [1, 8]
]).astype(float)

r = np.array([
    1, 2, 3
]).astype(float)

compare(J, r)

# Вырожденная матрица
J = np.array([
    [1, 2],
    [0, 0],
    [0, 0]
]).astype(float)

r = np.array([
    1, 2, 3
]).astype(float)

compare(J, r)

# Крайне близкая матрица к вырожденной
J = np.array([
    [1, 2],
    [0, 0],
    [0, 1e-10]
]).astype(float)

r = np.array([
    1, 2, 3
]).astype(float)

compare(J, r)

J = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [4, 6, 9]
]).astype(float)

r = np.array([
    8, 9, 10
]).astype(float)

compare(J, r)

rng = np.random.default_rng(seed=42)


def generate_y(X, relation, true_params, noise):
    m, n = X.shape
    y = np.zeros(m)
    for i in range(m):
        y[i] = relation(X[i], true_params) + rng.normal(loc=0.0, scale=noise)

    return y


def func(value, params):
    return value[0] ** 2 * params[2] + value[0] * params[1] + params[0]


def dfunc(value, params):
    return np.array([
        1.0,
        value[0],
        value[0] ** 2
    ])


pars = np.array([-5.0, -5.0, 3.0])
X = np.linspace(-3.0, 3.0, 50).reshape(-1, 1)
y = generate_y(X, func, pars, 10)  # Большой шум, просто чтобы графики были визуально разделимы

initial = np.array([1.0, 1.0, 1.0])
est = gauss_newton(X, y, func, dfunc, initial, lstsq_SVD, 1)
print("Estimated =", est)
print("True =", pars)
print("Difference norm =", np.linalg.norm(est - pars))

ox = np.linspace(-4.0, 4.0, 1000)
f1 = np.array([func(np.array([t]), pars) for t in ox])
f2 = np.array([func(np.array([t]), est) for t in ox])
plt.figure(figsize=(9, 5))
plt.scatter(X, y, s=4, c='black', label="Данные")
plt.plot(ox, f1, label="Истинная функция")
plt.plot(ox, f2, label="Построенная функция (SVD)")
plt.legend()
plt.grid(True)
plt.show()
