import numpy as np
import matplotlib.pyplot as plt
from fopdtUtils import Fopdt

class SimulationUtils:
    """
    用于控制器调试和仿真测试的工具类
    1. 包含生成设定值生成器；
    2. cv测试信号模拟生成器，该生成器可以包含可调的扰动项；
    3. 可以同时生成多个cv测试信号
    """

    def __init__(self):
        self.cv_functions = {}

    def add_cv_function(self, name, fopdt):
        """
        向仿真工具的实例中增加一个cv的一阶响应函数
        :param name: cv_function的名字
        :param fopdt: class FOPDT.
        :return: none
        """
        self.cv_functions[name] = fopdt

    @staticmethod
    def create_sps(sp_numbers, times, sps):
        """
        创建一组设定值
        :param times: list. 设定值变化的时间点，len < n
        :param sps: list. 设定值的值, len < n
        :param sp_numbers: int. 整个时间序列的长度, len = n
        :return: res: np.array. 一组由设定值组成的二维数组. shape = n*2
        """
        t_sequence = np.linspace(0, sp_numbers-1, sp_numbers)
        sp_sequence = np.zeros(sp_numbers)
        res = np.dstack((t_sequence, sp_sequence)).reshape((sp_numbers, 2))

        sp_index = 0
        t_index = 0
        for t in range(0, sp_numbers):
            if t < times[t_index]:
                res[t][1] = sps[sp_index]
            else:
                sp_index += 1
                t_index += 1
                if t_index > len(times) - 1:
                    t_index = len(times) - 1
                    sp_index = len(sps) - 1
                res[t][1] = sps[sp_index]
        return res

    @staticmethod
    def plot_sps(time_sp_array):
        """
        绘制设定值随时间变化的曲线(只是方便create_sps()方法而已，其他没啥用)
        :param time_sp_array: np.array, shape(n,2).
        :return: none
        """
        sps = time_sp_array[..., 1]
        times = time_sp_array[..., 0]
        plt.plot(times, sps, 'rx', markersize=10)
        plt.plot(times, sps, 'r-', drawstyle='steps-post')
        ax = plt.gca()
        ax.set_xlim(0, len(times)-1)
        plt.grid()
        plt.show()

    def get_next_cv_value(self, name, uv):
        """
        获取当前实例某个cv函数的下一个值
        :param name: cv函数的名字
        :param uv: uv
        :return: cv测量值
        """
        fopdt = self.cv_functions.get(name)
        return fopdt.next(uv)
