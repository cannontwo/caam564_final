import linear_model
import numpy as np
import viz_1_1d
import random

# Plotting stuff
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

for i in range(10):
    a = random.randint(-10, 10)
    b = random.randint(-10, 10)
    model = linear_model.LinearModel(1, 1)
    model.A = np.array([[a]])
    model.B = np.array([[b]])
    model.limits_low = np.array([-10.0])
    model.limits_high = np.array([10.0])

    controller = model.create_lqr_controller(np.array([0.0]))
    controller.limits_low = np.array([-2.0])
    controller.limits_high = np.array([2.0])

    viz_1_1d.plot_1_1d_control_traj(model.predict_state,
            controller.get_optimal_action, title="a = {}, b = {}".format(a, b),
            time=10.0)

    viz_1_1d.plot_1_1d_model(model, action_limit_low=controller.limits_low[0], action_limit_high=controller.limits_high[0])
