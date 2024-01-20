import time

import numpy as np
from scipy.optimize import approx_fprime, minimize
from scipy.spatial import ConvexHull
from scipy.special import expit


"""
the final result is [0.70725779 0.70725779],the funtion value is[-1.41408841 -0.69871439],the iteration number is 415
Elapsed time: 0.6405811309814453 seconds

"""

def f1(x):
    a = (-x[0] - x[1])
    b = (-x[0] - x[1] + x[0]**2 + x[1]**2 -1)
    res = np.max([a,b])
    return res


def f2(x):
    a = x[0] ** 2 + x[1] ** 2 - 1
    res = -x[0] + 20 * np.maximum(a,0)
    return res





def f(x):
    return np.array([f1(x),f2(x)])

def grad_f1(x):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f1_wrapper(x):
        return f1(x)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f1_wrapper, epsilon)
    return grad

def grad_f2(x):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f2_wrapper(x):
        return f2(x)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f2_wrapper, epsilon)
    return grad

def grad_f(x):
    return np.array([grad_f1(x),grad_f2(x)])

def h(x,t,v,c):
    res = f(x+t*v) - f(x) + c* t* ((np.linalg.norm(v))**2)
    return res

def subgradient(x, v, var_e, c, j):
    a = 0
    b = var_e / (np.linalg.norm(v))
    # print(f'b={b}')
    t = (a + b) / 2
    while True:
        # print(f'正在计算term项')
        xi = grad_f(x + t * v)[j]
        term1 = np.dot(v, xi)
        term2 = -c * ((np.linalg.norm(v)) ** 2)
        print(f'term1={term1},term2={term2}')
        if term1 > term2:
            # print(f'need break')
            break
        if h(x, b, v, c)[j] > h(x, t, v, c)[j]:
            a = t
        else:
            b = t
        t = (a + b) / 2
        # print(f't={t}')
    return t, xi

def solve_convex_optimization(xi_matrix):
    # 定义目标函数
    def objective_function(lambda_vector):
        xi_sum = np.dot(lambda_vector, xi_matrix)
        return np.linalg.norm(xi_sum)**2

    # 定义约束条件
    def constraint(lambda_vector):
        return np.sum(lambda_vector) - 1

    # 定义拉格朗日函数
    def lagrangian(lambda_vector, alpha, beta_vector):
        return objective_function(lambda_vector) + alpha * constraint(lambda_vector) - np.dot(beta_vector, lambda_vector)

    # 初始参数
    lambda_vector_guess = np.ones(len(xi_matrix)) / len(xi_matrix)  # 初始猜测，均匀分布
    alpha_guess = 0.0
    beta_vector_guess = np.zeros_like(lambda_vector_guess)

    # 定义边界，确保 lambda 在 (0, 1) 之间
    bounds = [(0, 1) for _ in range(len(lambda_vector_guess))]

    # 最小化拉格朗日函数
    result = minimize(
        fun=lambda lambda_vector: lagrangian(lambda_vector, alpha_guess, beta_vector_guess),
        x0=lambda_vector_guess,
        constraints={'type': 'eq', 'fun': constraint},
        bounds=bounds
    )

    # 提取结果
    optimal_lambda = result.x
    optimal_xi = np.dot(optimal_lambda, xi_matrix)

    return optimal_lambda, optimal_xi

def search_direction(x, var_e, delta, c):
    W = np.array([-grad_f(x)[0],-grad_f(x)[1]])  # Initialize W with the gradient of f at x

    Il = []
    while True:
        optimal_lambda, optimal_xi = solve_convex_optimization(W)
        v = optimal_xi


        if np.linalg.norm(v) <= delta:
            break

        for j in range(1):
            if np.all(f(x + (var_e / np.linalg.norm(v)) * v) > (f(x) - c * var_e * np.linalg.norm(v))[j]):
                Il.append(j)
        if not Il:
            break

        for j in Il:
            t, xi = subgradient(x, v, var_e, c, j)
            W = np.vstack([W, -xi])

    return v


def solver(f, t0, v, x):
    s = 0
    while True:
        term1 = f(x + 2 ** (-s) * t0 * v)
        term2 = f(x) - 2 ** (-s) * t0 * ((np.linalg.norm(v)) ** 2)
        if all(term1 <= term2):
            return s
        else:
            s += 1

def main(x,var_e,delta,c,t0):
    start_time = time.time()
    j = 1
    while True:
        v = search_direction(x,var_e,delta,c)
        print(f'v={v}')
        if np.linalg.norm(v) <= delta:
            break
        s = solver(f, t0, v, x)
        t = max([2**(-s) * t0, var_e/(np.linalg.norm(v))])
        x += t * v
        j += 1
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算程序运行时间
    return x,j,elapsed_time

x = np.array([1.0, 1.0])  # Define x as a float64 array

res,j,elapsed_time = main(x,var_e=1e-3,delta=1e-3,c=0.25,t0=1)
f11 = f1(res)
f22 = f2(res)
value = np.array([f11,f22])
print(f'the final result is {res},the funtion value is{value},the iteration number is {j}')
print(f'Elapsed time: {elapsed_time} seconds')