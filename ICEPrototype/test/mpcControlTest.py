import unittest
from mpcControl import MPC
from fopdtUtils import Fopdt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):

    def test_mpc_control(self):
        init_uvs = np.array([3, 4, 5])
        set_points = np.array([5, 5, 5])
        cv_init = 1
        uv_init = 2
        k = 3
        tau = 5
        disturb = 0
        uv_limit = 2
        circle = 5
        predict_horizon = 4
        control_horizon = 2
        sp_y_wight = 10
        uv_step_weight = 20

        fopdt = Fopdt(k, tau, cv_init, disturb)
        config = {
            "fopdt": fopdt,
            "predict_horizon": predict_horizon,
            "control_horizon": control_horizon,
            "cv_init": cv_init,
            "uv_init": uv_init,
            "circle": circle,
            "uv_limit": uv_limit,
            "set_points": set_points,
            "sp_y_wight": sp_y_wight,
            "uv_step_weight": uv_step_weight
        }

        # 测试损失函数计算
        init_cost = MPC.cost_function(
            init_uvs,
            config
        )

        # 测试最优解
        solution = MPC.optimize_solution(
            init_uvs,
            config
        )

        mini_uvs = solution.x
        mini_cost = MPC.cost_function(
            mini_uvs,
            config
        )

        print("init cost: %f, mini cost: %f" % (init_cost, mini_cost))
        print("init uvs: %s, mini uvs: %s" % (init_uvs, mini_uvs))

if __name__ == '__main__':
    unittest.main()
