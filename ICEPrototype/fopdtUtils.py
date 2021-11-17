import numpy as np
from scipy.integrate import odeint
import random
from collections import deque
import warnings


class _UVStore:
    """
    以栈的形式，储存uv的历史数据，储存元素的上限一般等于dead time
    """

    class UVInfo:
        """
        UV数据的实体类
        """
        def __init__(self, time, value):
            self.time = time
            self.value = value

        def get_value(self):
            return self.value

        def get_time(self):
            return self.time

    def __init__(self, max_len):
        self.stack = deque(maxlen=max_len)

    def save_uv(self, time, value):
        self.stack.append(self.UVInfo(time, value))

    def pop_uv(self):
        if len(self.stack) == 0:
            warnings.warn('空的UV储存器')
            return 0
        else:
            return self.stack[0].get_value()


class Fopdt:

    def __init__(self, k, tau, y0, dv, theta):
        """
        构造函数
        :param k: float. process gain
        :param tau: float. process time constant
        :param y0: float. t = 0时的cv值
        :param dv: float. 扰动幅度. (eg. dv = 5, 则next方法返回的cv值会上下随机波动5%, dv = 0则无波动)
        """
        # 增益
        self.k = k

        # 时间常数
        self.tau = tau

        # 迟滞时间
        self.theta = theta

        # 当前实例的t = 0时的cv值
        self.y0 = y0

        # 当前实例的时间步
        self.t = 0

        # 扰动幅度. float, dv > 0
        self.dv = dv

        # 历史uv储存器
        self.uv_store = _UVStore(theta)

    def fopdt(self, y, t, u):
        """
        默认的一阶响应函数，用于简单的调试
        :param y: cv output, shape(n,)
        :param t: list. time list, shape(n,)
        :param u: list. uv list, shape(n,)
        :param k: float. process gain
        :param tau: float. process time constant
        :return: np.array. cv output, shape(n,)
        """
        dydt = (-y + self.k * u)/self.tau
        return dydt

    def get_fopdt_fn(self):
        """
        返回当前实例的fopdt调用
        """
        return self.fopdt

    def solve_sequence_fopdt(self, times, uvs, y0):
        """
        求解多个操作步的输出
        :param times:list. time list, shape(n,)
        :param uvs:list. uv list, shape(n,)
        :param y0: float. cv init condition
        :return: np.array. cv output, shape(n,)
        """
        if len(times) != len(uvs):
            raise Exception("时间数组和输入数组维度不一致!")
        else:
            length = len(times)
            y_res = np.zeros(length)
            y_res[0] = y0
            for i in range(1, length):
                t = times[i]
                self.uv_store.save_uv(t, uvs[i])
                uv = self.uv_store.pop_uv()
                y = self.solve_step_fopdt(times[i - 1], t, y0, uv)
                y_res[i] = y0 = y
            return y_res

    @staticmethod
    def solve_sequence_fopdt(fopdt, times, uvs, y0):
        """
        静态方法。求解多个操作步的输出
        :param fopdt: class FOPDT
        :param times:list. time list, shape(n,)
        :param uvs:list. uv list, shape(n,)
        :param y0: float. cv init condition
        :return: np.array. cv output, shape(n,)
        """
        if len(times) != len(uvs):
            raise Exception("时间数组和输入数组维度不一致!")
        else:
            length = len(times)
            y_res = np.zeros(length)
            y_res[0] = y0
            for i in range(1, length):
                t = times[i]
                y = odeint(fopdt.get_fopdt_fn(), y0, [times[i - 1], t], args=(uvs[i],))[1][0]
                y_res[i] = y0 = y
            return y_res

    def solve_step_fopdt(self, start_t, end_t, init_y, uv):
        """
        求解单个操作步的输出
        :param start_t: float. end time
        :param end_t: float. start time
        :param init_y: float. output at time = start_t
        :param uv: float. uv
        :return: y. float. output at time = end_t
        """
        assert end_t > start_t, "end_t 需要大于 start_t!"
        t = [start_t, end_t]
        y = odeint(self.fopdt, init_y, t, args=(uv,))
        return y[1][0]

    def next(self, uv):
        """
        计算当前实例化对象，下一个时间步的输出
        :param uv: float. uv
        :return: y. float. 下一个时间步的输出
        """
        t_next = self.t + 1
        y = self.solve_step_fopdt(self.t, t_next, self.y0, uv)
        self.y0 = y
        self.t = t_next

        dv = (self.dv * (2 * random.random() - 1))/100
        return y * (1 + dv)


