import numpy as np
from linear_model import LinearModel
from one_one_d import OneOneDSystem

# Plotting stuff
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def stupid_system(s, u):
    """
    Get single-step update of system.
    """
    a = 1.0
    b = 1.0
    return a * s + b * u

def get_stupid_control(s, s_ref=0.0, k=1.0):
    """
    Get control for current state s and gain k.
    """
    return -1.0 * k * (s - s_ref)

def plot_1_1d_control_traj(system_func, control_func, time=1.0, s_0=1.0,
        title="", action_abs_max=10.0):
    t = np.linspace(0.0, time)

    s = np.zeros_like(t)
    s[0] = s_0

    plt.figure()

    for i in range(len(t) - 1):
        array_s = np.array([s[i]])
        direction = 1.0 if control_func(array_s)[0] > 0 else -1.0
        s[i+1] = system_func(array_s, control_func(array_s))
        plt.arrow(t[i], s[i], 0.0, 0.1*direction,
                head_width=0.01, head_length=0.01,
                alpha=float(abs(control_func(array_s)[0]))/action_abs_max)

    plt.plot(t, s)
    plt.title(title)
    plt.show()


def plot_1_1d_model(model, action_limit_low=-1, action_limit_high=1, ax=None, save=False, save_loc="/tmp/tmp.png"):
    """
    Plot a linear model as a surface in 3D along the state and action axes specified. 

    Args:
        model: The 1-1D model to plot.
        action_limit_low: Action lower limits for plotting.
        action_limit_high: Action upper limits for plotting.
        ax: Axis on which to plot; otherwise plots independently. If not None,
            should be set up for 3D plotting.
        save: Whether to save the plot.
        save_loc: Where to save the plot.

    Returns:
        None
    """
    assert(isinstance(model, LinearModel))

    if ax is None:
        fig = plt.figure()
        using_ax = fig.gca(projection='3d')
    else:
        using_ax = ax

    
    X = np.arange(model.limits_low[0], model.limits_high[0], 0.25)
    Y = np.arange(action_limit_low, action_limit_high, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[model.predict_state(np.array([X[i, j]]), np.array([Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    surf = using_ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    using_ax.set_zlim(model.limits_low[0], model.limits_high[0])
    using_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    using_ax.set_xlabel("State")
    using_ax.set_ylabel("Action")
    using_ax.set_zlabel("Next State")

    #using_ax.colorbar(surf, shrink=0.5, aspect=5)

    if save and ax is None:
        plt.savefig(save_loc)
    elif ax is None:
        plt.show()

def plot_1_1d_system(system, ax=None, save=False, save_loc="/tmp/tmp.png"):
    """
    Plot a 1-1D system as a surface in 3D along the state and action axes specified. 

    Args:
        system: The 1-1D system to plot.
        ax: Axis on which to plot; otherwise plots independently. If not None,
            should be set up for 3D plotting.
        save: Whether to save the plot.
        save_loc: Where to save the plot.

    Returns:
        None
    """
    assert(isinstance(system, OneOneDSystem))

    if ax is None:
        fig = plt.figure()
        using_ax = fig.gca(projection='3d')
    else:
        using_ax = ax

    def get_next_state(state, action):
        system.reset(state)
        _, _, next_state, _ = system.step(action)
        return next_state
    
    X = np.arange(system.state_limits[0], system.state_limits[1], 0.25)
    Y = np.arange(system.action_limits[0], system.action_limits[1], 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[get_next_state(X[i, j], Y[i, j]) for j in range(X.shape[1])] for i in range(X.shape[0])])

    system.reset()

    surf = using_ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    using_ax.set_zlim(system.state_limits[0], system.state_limits[1])
    using_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    using_ax.set_xlabel("State")
    using_ax.set_ylabel("Action")
    using_ax.set_zlabel("Next State")

    #using_ax.colorbar(surf, shrink=0.5, aspect=5)

    if save and ax is None:
        plt.savefig(save_loc)
    elif ax is None:
        plt.show()

if __name__ == "__main__":
    plot_1_1d_control_traj(stupid_system, get_stupid_control)
