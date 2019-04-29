import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import PolynomialFeatures

class OneOneDSystem:
    """
    Class representing a discrete time dynamical system in one state dimension
    and one action dimension (1-1D). These systems are nice because it is easy
    to define and visualize versions of different order, and because they are
    the most basic form of the kind of system that RL algorithms seek to
    control. It is assumed that states and actions are real numbers.
    """

    def __init__(self, update_func, reward_func, s_0=1.0,
            state_limits=[-2.0,2.0], action_limits=[-1.0, 1.0]):
        """
        Initialize the 1-1D system.

        Args:
            update_func: The update function defining the dynamical system, 
                         i.e. s' = update_func(s, a).
            reward_func: The reward function for this dynamical system. Should be
                         bounded and have call signature r = reward_func(s, a, s')
            s_0: The start state for the system.
            state_limits: The lower and upper limits for the state space.
            action_limits: The lower and upper limits for the action space.

        Returns:
            None
        """
        assert(len(state_limits) == 2)
        assert(len(action_limits) == 2)

        self.s = s_0
        self.s_0 = s_0
        self.update_func = update_func
        self.reward_func = reward_func
        self.state_limits = state_limits
        self.action_limits = action_limits

    def reset(self, s_0=None):
        """
        Reset the state of the system.

        Args:
            s_0: If not None, the state to reset to.

        Returns:
            The reset state.
        """
        if s_0 is not None:
            self.s = s_0
        else:
            self.s = self.s_0

        return self.s

    def step(self, a):
        """
        Step the 1-1D system represented by this object with the input action,
        or a clipped version according to the system limits.

        Args: 
            a: The action to take.

        Returns:
            A tuple of (prev_state, limited_action, new_state, reward).
        """
        limited_action = min(self.action_limits[1], max(self.action_limits[0], a))

        prev_state = self.s
        new_state = min(self.state_limits[1], max(self.state_limits[0], self.update_func(prev_state, a)))
        reward = self.reward_func(prev_state, a, new_state)
        self.s = new_state
    
        return (prev_state, limited_action, new_state, reward)

    def sim_traj(self, control_func, time_steps=100):
        """
        Simulate this system under the input control function for the input
        number of time steps, starting from the current state. 

        Args:
            control_func: The control function to apply to the system while
                          simulating.
            time_steps: The number of times to simulate for.

        Returns:
            A tuple of (states, actions, rewards), where each element has
            length equal to time_steps.
        """
        states = [self.s]
        a_0 = control_func(self.s)
        _, real_a_0, _, r_0 = self.step(a_0)

        actions = [real_a_0]
        rewards = [r_0]

        for i in range(time_steps - 1):
            states.append(self.s)
            _, a, _, r = self.step(control_func(self.s))

            actions.append(a)
            rewards.append(r)

        return (states, actions, rewards)

    def plot_traj(self, control_func, arrow_scale=1.0, time_steps=100, ax=None, save=False, save_loc="/tmp/tmp.png", title=""):
        """ 
        Plot the evolution of this system in state and reward space over a
        specified number of time_steps according to some control function.

        Args:
            control_func: A function specifying an action for each state.
            arrow_scale: Amount to scale action magnitude by for arrow plotting.
            time_steps: The number of timesteps to plot.
            ax: Axis to plot on, if not None.
            save: Whether to save the generated plot.
            save_loc: Where to savethe generated plot.
            title: Title for the generated plot.

        Returns:
            None
        """
        states, actions, rewards = self.sim_traj(control_func, time_steps=time_steps)
        t = np.linspace(0.0, time_steps, num=time_steps)
        
        if ax is None:
            fig = plt.figure(figsize=(8, 6), dpi=200)
            using_ax = fig.gca()
        else:
            using_ax = ax

        cmap = matplotlib.cm.coolwarm
        norm = matplotlib.colors.Normalize(vmin=self.action_limits[0], vmax=self.action_limits[1])

        for i in range(time_steps):
            direction = 1.0 if actions[i] > 0 else -1.0
            using_ax.arrow(t[i], states[i], 0.0, arrow_scale*direction,
                    head_width=0.4, head_length=0.1, color=cmap(norm(actions[i])))

        #plt.colorbar(ax=using_ax, cmap='Spectral')
        cb_axis = using_ax.inset_axes([0.57, 0.9, 0.4, 0.1])
        cb1 = matplotlib.colorbar.ColorbarBase(cb_axis, cmap=cmap, norm=norm,
                orientation='horizontal')

        using_ax.plot(t, states, color="black")
        using_ax.set_ylim(self.state_limits[0] - 3, self.state_limits[1] + 3)
        using_ax.set_title(title)
        using_ax.set_xlabel('Timestep')
        using_ax.set_ylabel('State/Action')

        if not save and ax is None:
            plt.show()
        elif ax is None:
            plt.savefig(save_loc)

def make_random_poly_system(n, m, rand_limits=[-1, 1]):
    """
    Construct a 1-1D system with an order n polynomial update function and
    order m polynomial reward function, both with random coefficients. 

    Args:
        n: The order of the system's update function.
        m: The order fo the systems's reward function.
        rand_limits: Limits for coefficients defining the system.

    Returns:
        The constructed 1-1D system, update function coefficients, and reward
        function coefficents.
    """
    assert(len(rand_limits) == 2)

    update_coeffs = [random.uniform(rand_limits[0], rand_limits[1]) for i in range(2*n)]
    print("Update coefficients are:")
    print("\t{}".format(update_coeffs))

    reward_coeffs = [random.uniform(rand_limits[0], rand_limits[1]) for i in range(3*m)]
    print("Reward coefficients are:")
    print("\t{}".format(reward_coeffs))

    # TODO : Fix to include cross terms; use poly.fit_transform([[s,a]])[0][1:].
    update_poly = PolynomialFeatures(n)
    blank_update_powers = update_poly.fit_transform(np.zeros((1, 2)))[0][1:]

    reward_poly = PolynomialFeatures(m)
    blank_reward_powers = reward_poly.fit_transform(np.zeros((1, 3)))[0][1:]
    
    update_coeffs = np.random.uniform(low=rand_limits[0], high=rand_limits[1], size=blank_update_powers.shape)
    reward_coeffs = np.random.uniform(low=rand_limits[0], high=rand_limits[1], size=blank_reward_powers.shape)

    def update_func(s, a):
        z = np.array([[s, a]])
        powers = update_poly.fit_transform(z)[0][1:]

        return np.sum(np.multiply(powers, update_coeffs))

    def reward_func(s, a, s_prime):
        z = np.array([[s, a, s_prime]])
        powers = reward_poly.fit_transform(z)[0][1:]

        return np.sum(np.multiply(powers, reward_coeffs))

    return (OneOneDSystem(update_func, reward_func), update_coeffs, reward_coeffs)

def make_random_harmonic_system(rand_limits=[-1, 1]):
    """
    Construct a 1-1D system with random harmonic (linear combination of sine
    and cosine) update and reward functions.

    Args:
        rand_limits: Limits for coefficients defining the system.

    Returns:
        The constructed 1-1D system.
    """
    # TODO

    pass
