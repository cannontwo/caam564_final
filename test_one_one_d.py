import one_one_d
import linear_model

import numpy as np


for i in range(5):
    print("Plotting system {}".format(i))
    system, update_coeffs, reward_coeffs = one_one_d.make_random_poly_system(1, 1)
    print(one_one_d.get_system_equation(update_coeffs))
    model = linear_model.LinearModel(1, 1)
    model.A = np.array([[update_coeffs[0]]])
    model.B = np.array([[update_coeffs[1]]])

    controller = model.create_lqr_controller(np.array([0.0]))
    remapped_control = lambda s: controller.get_optimal_action(np.array([s]))[0]
    states, actions, rewards = system.sim_traj(remapped_control)

    title = one_one_d.get_system_equation(update_coeffs)

    system.reset()
    system.plot_traj(remapped_control, arrow_scale=1.0, save=True, save_loc="/home/cannon/Documents/comp590/plots/random_1_1_system_{}.png".format(i), title=title)
