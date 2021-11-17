import numpy as np
from fopdtUtils import Fopdt
from scipy.optimize import minimize


class MPC:

    # 1. 在k时刻测量/估计系统当前的状态值yk


    # 2. 从k时刻开始(包括k时刻)，以时间间隔i. 给予系统如下输入
    # [k ~ k + i], u0
    # [k+i ~ k+2i], u1
    # [k+2i ~ k+3i], u2

    @staticmethod
    def cost_function(uvs, arg_dict):
        """
        根据uvs和mpc的配置，计算cost
        :param uvs: np.array. 整个预测区间上的操作变量数组, shape(ph, n)
        :param arg_dict = {
            model: function. 预测模型，比如FOPDT
            ph: int. 预测区间长度
            ch: int. 控制区间长度
            cv0: float. y的初始值
            uv0: float. uv的初始值
            circle: float. 控制周期
            u_max_move: float. u的最大变化范围
            sps: float. 当前预测区间的设定值
            sp_cv_wight: float. 目标值和设定值误差的残差权重
            uv_step_weight: float. uv增量的残差权重
        }

        :return: cost: float. 残差
        """

        model, ph, ch, cv0, uv0, circle, u_max_move, sp, sp_cv_wight, uv_step_weight = arg_dict.values()

        cost = np.zeros(ph)
        before_uv = uv0
        before_cv = cv0

        # 根据控制区间的长度处理uvs,超出控制区间的uv统一设置未控制区间最后时刻的uv
        uvs[ch:] = uvs[ch]

        # 遍历sps, i迭代次数，sp
        for i in range(0, ph):
            # print("i = %i, sp = %f" % (i, set_point))
            # 基于输入u和预测模型model，计算预测区间p内的输出
            next_uv = uvs[i]
            start_time = i * circle
            end_time = (i + 1) * circle
            next_cv = model.solve_step_fopdt(start_time, end_time, before_cv, next_uv)

            # 计算这次迭代的损失
            delta_uv = next_uv - before_uv
            cost[i] = sp_cv_wight * ((sp - next_cv) ** 2) + uv_step_weight * (delta_uv ** 2)

            # 更新uv, cv
            before_cv = next_uv
            before_uv = next_cv

        sum_cost = np.sum(cost)
        return sum_cost

    @staticmethod
    def optimize_solution(init_uvs, arg_dict):
        """
        给定uvs的初始猜测值和mpc配置，计算最优uvs
        :param init_uvs. np.array. 整个预测区间上的操作变量数组
        :param arg_dict = {
            model: function. 预测模型，比如FOPDT
            ph: int. 预测区间长度
            ch: int. 控制区间长度
            cv0: float. y的初始值
            uv0: float. uv的初始值
            circle: float. 控制周期
            u_max_move: float. u的最大变化范围
            sps: np.array. 整个预测区间y的设定值, shape(ph, n)
            sp_cv_wight: float. 目标值和设定值误差的残差权重
            uv_step_weight: float. uv增量的残差权重
        }
        :return: solution: 求解结果
        """
        solution = minimize(
            MPC.cost_function,
            init_uvs,
            args=arg_dict,
            method='SLSQP'
        )
        return solution

    @staticmethod
    def filter_uv_by_max_move(uv0, uvs, uv_max_move):
        """
        根据uv的max move,获取一组新的uv
        :param uv0: uv的初值
        :param uvs: 待过滤的uv数组
        :param uv_max_move: uv最大移动步长
        :return:
        """
        filter_uvs = []

        # 控制uvs[0]的步幅
        if np.abs(uvs[0] - uv0) <= uv_max_move:
            filter_uvs.append(uvs[0])
        else:
            filter_uvs.append(uvs[0] + uv_max_move * (uvs[0] / np.abs(uvs[0])))

        # 控制其他uv的步幅
        for i in uvs[1:]:
            uv = uvs[i]
            if np.abs(uvs[i] - uvs[i-1]) <= uv_max_move:
                filter_uvs.append(uv)
            else:
                # uv = uv + (uv/|uv|) * uv_max_move
                filter_uvs.append(uv + uv_max_move * (uv/np.abs(uv)))

        return filter_uvs

