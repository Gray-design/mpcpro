import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

# Define process model
def process_model(y, t, u, K, tau):
    # arguments
    #  y   = outputs
    #  t   = time
    #  u   = input value
    #  K   = process gain
    #  tau = process time constant

    # calculate derivative
    dydt = (-y + K * u) / (tau)

    return dydt

# Define Objective function
# 记，中心为i，宽度为P的邻域，为邻域[i]，左邻域为[-i]，右邻域为[i-],整个邻域包含2P+1个元素
# 记，邻域[i]从左至右的第k个元素为[i(k)]
# 记，[i(k)]对应于时间轴上的值为[i(k)-t]，若[i(k)-t]]>0则称[i(k)]有意义
def objective(u_hat):
    # Prediction
    for k in range(1, 2 * P + 1):
        if k == 1:
            y_hat0 = yp[i - P]

        if k <= P:
            # i-P+k<0，表示[-i]的第一个元素对应的时刻 < 0；那么mv==0
            if i - P + k < 0:
                u_hat[k] = 0

            # 否则[i(k)]有意义，则
            else:
                u_hat[k] = u[i - P + k]

        elif k > P + M:
            u_hat[k] = u_hat[P + M]

        ts_hat = [delta_t_hat * (k - 1), delta_t_hat * (k)]
        y_hat = odeint(process_model, y_hat0, ts_hat, args=(u_hat[k], K, tau))
        y_hat0 = y_hat[-1]
        yp_hat[k] = y_hat[0]

        # Squared Error calculation
        sp_hat[k] = sp[i]

        # 创建变量，用于储存mv每步的变化
        delta_u_hat = np.zeros(2 * P + 1)

        if k > P:
            # Ui - Ui-1
            delta_u_hat[k] = u_hat[k] - u_hat[k - 1]

            # se = (sp - yp)^2 + 20 * (Ui - Ui-1)^2
            se[k] = (sp_hat[k] - yp_hat[k]) ** 2 + 20 * (delta_u_hat[k]) ** 2

    # Sum of Squared Error calculation
    obj = np.sum(se[P + 1:])
    return obj

# FOPDT Parameters
K = 3.0  # gain
tau = 5.0  # time constant
ns = 100  # Simulation Length
t = np.linspace(0, ns, ns + 1)
delta_t = t[1] - t[0]

# Define horizons
P = 30  # Prediction Horizon
M = 10  # Control Horizon

# Input Sequence
# 保存所有操作变量的向量
u = np.zeros(ns + 1)

# Setpoint Sequence
# 定义setpoint sequence
sp = np.zeros(ns + 1 + 2 * P)
sp[10:40] = 5
sp[40:80] = 10
sp[80:] = 3
# Controller setting
maxmove = 1

## Process simulation
yp = np.zeros(ns + 1)

#  Create plot
plt.figure(figsize=(10, 6))
plt.ion()
plt.show()

for i in range(1, ns + 1):
    if i == 1:
        y0 = 0
    # --------------------------------------获取CV的测量值-----------------------------------------
    # 基于上一个测量值和时间增量，计算新的cv测量值
    ts = [delta_t * (i - 1), delta_t * i]
    y = odeint(process_model, y0, ts, args=(u[i], K, tau))

    # 记录最新的测量值
    y0 = y[-1]

    # 将测得的所有cv保存在一个向量中
    yp[i] = y[0]

    # ----------------------------------- 构建损失函数所需的变量--------------------------------------
    # Declare the variables in fuctions
    # 当前预测区间的时间窗口。可视为中心为i，宽度为P的邻域
    t_hat = np.linspace(i - P, i + P, 2 * P + 1)

    # 时间增量(每次都等于1？）
    delta_t_hat = t_hat[1] - t_hat[0]

    # 误差向量，为了方便起见，维度和t_hat一致
    se = np.zeros(2 * P + 1)
    # cv预测值向量，为了方便起见，维度和t_hat一致
    yp_hat = np.zeros(2 * P + 1)

    # 下一步mv预测值(为什么他要这么大的维度？)
    u_hat0 = np.zeros(2 * P + 1)

    # 预测区间内的设定值？
    sp_hat = np.zeros(2 * P + 1)

    # 损失函数的值
    obj = 0.0

    # initial guesses
    # 生成所有操作变量的初值(类似于NN中初始化输入向量，应该生成P个初始的猜测值？)
    for k in range(1, 2 * P + 1):

        # 获取过去mv的值
        if k <= P:
            # 如果时间小于0，则所有mv=0
            if i - P + k < 0:
                u_hat0[k] = 0
            # 否则从mv储存变量中获取
            else:
                u_hat0[k] = u[i - P + k]
        # 生成未来mv的值(这里应该就是直接等于0吧？)
        elif k > P:
            u_hat0[k] = u[i]

    # show initial objective

    # 获取初始残差
    # print('Initial SSE Objective: ' + str(objective(u_hat0)))

    # MPC calculation
    # 执行MPC计算，求最优解
    start = time.time()
    solution = minimize(objective, u_hat0, method='SLSQP')
    u_hat = solution.x

    end = time.time()
    elapsed = end - start

    # 计算最优解的残差
    # print('Final SSE Objective: ' + str(objective(u_hat)))
    # print('Elapsed time: ' + str(elapsed) )

    # 计算相邻两个数的差，根据给定的mv最大单步变化重新调整mv的向量
    delta = np.diff(u_hat)
    if i < ns:
        if np.abs(delta[P]) >= maxmove:
            # 如果需要增大mv，则i+1时刻，向下写入的mv = ui + maxmove
            if delta[P] > 0:
                u[i + 1] = u[i] + maxmove
            # 如果需要减小mv，则i+1时刻，向下写入的mv = ui - maxmove
            else:
                u[i + 1] = u[i] - maxmove
        # 否则向下写入的mv = ui + delta
        else:
            u[i + 1] = u[i] + delta[P]

    # plotting for forced prediction
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(t[0:i + 1], sp[0:i + 1], 'r-', linewidth=2, label='SP')
    plt.plot(t_hat[P:], sp_hat[P:], 'y-*', linewidth=2, label='Predicted Horizon SP')
    plt.plot(t[0:i + 1], yp[0:i + 1], 'k-', linewidth=2, label='Measured CV')
    plt.plot(t_hat[P:], yp_hat[P:], 'y--', linewidth=2, label='Predicted CV')
    plt.axvline(x=i, color='gray', alpha=0.5)
    plt.axis([0, ns + P, 0, 12])
    plt.ylabel('y(t)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.step(t[0:i + 1], u[0:i + 1], 'b-', linewidth=2, label='Measured MV')
    plt.plot(t_hat[P:], u_hat[P:], 'b.-', linewidth=2, label='Predicted MV')
    plt.axvline(x=i, color='gray', alpha=0.5)
    plt.ylabel('u(t)')
    plt.xlabel('time')
    plt.axis([0, ns + P, 0, 6])
    plt.legend()
    plt.draw()
    plt.pause(0.1)
