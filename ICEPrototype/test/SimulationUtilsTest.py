import unittest
from simulationUtils import SimulationUtils
import unittest
import numpy as np
from fopdtUtils import Fopdt
from mpcControl import MPC
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class SimulationUtilsTest(unittest.TestCase):

    def test_create_sps(self):
        """
        总共产生10个数据点，时间增量默认为1。
        t = [0, 2), sp = 3
        t = [2, 5), sp = 4
        t = [5, 9), sp = 2
        :return:
        """
        time_sp_array = SimulationUtils.create_sps(10, [2, 5], [3, 4, 2])
        SimulationUtils.plot_sps(time_sp_array)

    def test_mpc_simulation(self):

        # -----------------全局变量配置-----------------

        plt.ion()

        # 总步数
        total_steps = 60
        # 迭代周期
        circle = 1
        # 控制变量初值
        cv_init = 20

        # -----------------预测模型配置-----------------
        # 增益
        k = 3
        # 时间常数
        tau = 5
        # 扰动幅度
        disturb = 0
        # 预测模型迟滞
        theta_predict = 0
        # process model迟滞
        theta_process = 0
        # 测量值数据模型
        measure_fopdt = Fopdt(k, tau, cv_init, disturb, theta_process)

        # ------------------设定值配置-----------------
        time_keyframes = [15, 40]
        sp_keyframes = [6, 10, 4]
        # 整个区间上的设定值
        all_sps = SimulationUtils.create_sps(total_steps, time_keyframes, sp_keyframes)

        # -----------------MPC控制器配置-----------------
        predict_horizon = 6
        control_horizon = 3
        # 操作变量初值
        uv_init = 2
        # 操作变量的改变上下限
        uv_limit = 2
        # 第一个预测区间上，uv的初始值
        first_uvs = uv_init * np.random.rand(predict_horizon)
        # sp的预测值数组
        sps = all_sps[..., 1]
        # 设定值和操作变量残差权重
        sp_cv_wight = 20
        # 操作变量增量变化的残差权重
        uv_step_weight = 20
        # 预测模型
        predict_fopdt = Fopdt(k, tau, cv_init, 0, theta_predict)
        # mpc控制器求解参数
        config = {
            "fopdt": predict_fopdt,
            "predict_horizon": predict_horizon,
            "control_horizon": control_horizon,
            "cv_init": cv_init,
            "uv_init": uv_init,
            "circle": circle,
            "uv_limit": uv_limit,
            "set_points": 20,
            "sp_cv_wight": sp_cv_wight,
            "uv_step_weight": uv_step_weight
        }

        # print(first_uvs)
        # print(first_sps)

        # ----------------- Simulation -----------------

        # 用来画图的变量
        measure_cv_print = [cv_init]
        measure_cv_time_print = [0]
        history_uv_print = [uv_init]
        predict_cv_print = [cv_init]

        for i in range(0, total_steps):
            print("------------rolling optimize: %i----------" % i)

            # 在预测时域上执行二次规划
            solution = MPC.optimize_solution(
                first_uvs,
                config
            )

            if solution.success:
                # 向下写入uv
                write_uv = solution.x[0]
            else:
                write_uv = config.get("uv_init")

            # 计算下一步的测量值(实际项目中该值从数据采集系统中获得)
            measure_cv = measure_fopdt.next(write_uv)

            # 更新下一次计算的config
            config["cv_init"] = measure_cv
            config["uv_init"] = write_uv
            config["set_points"] = sps[i]

            print("write_uv = %f, measure_cv = %f, next_sps = %s" % (write_uv, measure_cv, config["set_points"]))

            # 添加数据到数组，用于画图
            measure_cv_print.append(measure_cv)
            measure_cv_time_print.append(i * circle)
            history_uv_print.append(write_uv)

            # 打印仿真结果
            plt.figure(figsize=(20, 5), )
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(1))
            plt.plot(measure_cv_time_print, measure_cv_print, 'r-', markersize=6, label="CV")
            plt.plot(all_sps[..., 0], all_sps[..., 1], 'b-', markersize=10, drawstyle='steps-post', label='SetPoint')
            plt.plot(measure_cv_time_print, history_uv_print, 'g-', drawstyle='steps-post', label='MV')
            ax = plt.gca()
            ax.set_xlim(0, total_steps - 1)
            plt.legend()
            plt.grid()
            plt.draw()
            plt.pause(0.1)


if __name__ == '__main__':
    unittest.main()
