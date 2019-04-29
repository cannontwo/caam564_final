import linear_model
import local_controllers
import one_one_d
import random
import viz_1_1d

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

n = 2

for i in range(10):
    fig = plt.figure(figsize=(12, 9), dpi=200)
    # Test fitting linear model to a 1-1D system of arbitrary order.
    system, update_coeffs, reward_coeffs = one_one_d.make_random_poly_system(n, 1)
    print("System has update coefficients:")
    print(update_coeffs)
    random_control = lambda s: random.uniform(system.action_limits[0], system.action_limits[1])

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("System Transition Function")
    viz_1_1d.plot_1_1d_system(system, ax=ax1)

    # Collect data by applying random actions in the system
    states, actions, rewards = system.sim_traj(random_control)
    states_prime = states[1:]
    states = states[:-1]

    # Matching expected format for model.fit()
    data = [(np.array([states[i]]), np.array([actions[i]]), rewards[i],
        np.array([states_prime[i]]), False) for i in range(len(states))]
    print("Fitting on {} datapoints".format(len(data)))

    # Fit linear model to data
    model = linear_model.LinearModel(1, 1)
    model.fit(data)
    model.limits_low = np.array([system.state_limits[0]])
    model.limits_high = np.array([system.state_limits[1]])
    print("Model is A={}, B={} (not fitting reward function now)\n".format(model.A, model.B))

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("Linear Model of Transition Function")
    viz_1_1d.plot_1_1d_model(model, ax=ax2, action_limit_low=system.action_limits[0], action_limit_high=system.action_limits[1])


    # Compute optimal controller for fit model
    system.reset()
    controller = model.create_lqr_controller(np.array([0.0]))
    remapped_control = lambda s: controller.get_optimal_action(np.array([s]))[0]

    # Plot evolution of system under linear control of fit system
    title = "Trajectory in {}th order system".format(n)

    ax3 = fig.add_subplot(2, 1, 2)
    system.plot_traj(remapped_control, ax=ax3, arrow_scale=1.0, title=title)

    plt.savefig('/home/cannon/Documents/caam564_final/plots/random_1_1_system_{}.png'.format(i))

