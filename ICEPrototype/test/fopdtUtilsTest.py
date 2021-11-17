import unittest
from fopdtUtils import Fopdt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.pyplot import MultipleLocator
from scipy.integrate import odeint
import numpy as np


class FopdtUtilsTest(unittest.TestCase):

    # demo函数 https://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    def demo_fopdt(self, y, t, uf, Km, taum, thetam):
        try:
            if (t - thetam) <= 0:
                um = 0
            else:
                um = uf(t - thetam)
        except:
            # print('Error with time extrapolation: ' + str(t))
            um = 0
        # calculate derivative
        dydt = (-y + Km * um) / taum
        return dydt

    # demo函数 https://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    def sim_model(self, ns, delta_t, Km, taum, thetam, uv_function):
        # input arguments
        # Km
        # taum
        # thetam
        # storage for model values
        ym = np.zeros(ns + 1)  # model
        # initial condition
        ym[0] = 0
        # loop through time steps
        for i in range(1, ns + 1):
            ts = [delta_t * (i - 1), delta_t * i]
            y1 = odeint(self.demo_fopdt, ym[i - 1], ts, args=(uv_function, Km, taum, thetam))
            ym[i] = y1[-1]
        return ym

    def test_solve_fopdt(self):
        """
        用于测试FOPDT函数。绘制不同条件下的y(t)图像
        :return:
        """
        y0 = 0
        uv0 = 0
        dv = 0
        k = 2.5
        tau = 3.0
        theta1 = 5
        theta2 = 1
        step_number = 30
        # u = [2,2,3,4,5,10,15,15,15]
        t = np.linspace(0, step_number, step_number+1)
        u = np.zeros(step_number + 1)
        u[5:] = 1.0

        fopdt1 = Fopdt(k, tau, y0, dv, theta1)
        fopdt2 = Fopdt(k, tau, y0, dv, theta2)
        y1 = fopdt1.solve_sequence_fopdt(t, u, y0)
        y2 = fopdt2.solve_sequence_fopdt(t, u, y0)

        # ---------------Demo的曲线-------------------
        theta3 = 4
        uf = interp1d(t, u)
        y3 = self.sim_model(step_number, 1, k, tau, theta3, uf)
        # -------------------------------------------

        # 设置图像尺寸
        plt.figure(figsize=(10, 5), )

        # 绘图
        plt.plot(t, y1, 'r-*')
        plt.plot(t, y2, 'b-*')

        # demo
        plt.plot(t, y3, 'y-*')

        # x轴的坐标刻度设置为1
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))

        # 设置x轴的范围
        plt.xlim(0, 30)

        # 显示网格
        plt.grid()
        plt.show()

    def test_solve_step_fopdt(self):
        y0 = 1
        k = 3
        tau = 5
        dv = 0
        theta = 5

        fopdt = Fopdt(k, tau, y0, dv, theta)
        y = fopdt.solve_step_fopdt(1, 2, y0, 2)
        print("test_solve_step_fopdt finished y = %f" % y)

    def test_next(self):
        y0 = 1
        k = 3
        tau = 5

        fopdt1 = Fopdt(k, tau, y0, 0)
        fopdt2 = Fopdt(k, tau, y0, 5)
        fopdt1.y0 = y0
        fopdt2.y0 = y0
        y1 = [y0]
        y2 = [y0]
        for i in range(0, 10):
            y1.append(fopdt1.next(2))
            y2.append(fopdt2.next(2))

        t = np.linspace(0, 10, 11)
        plt.plot(t, y1, "r--")
        plt.plot(t, y2, "-x")
        plt.show()


if __name__ == '__main__':
    unittest.main()
