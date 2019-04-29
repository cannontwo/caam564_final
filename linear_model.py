import numpy as np
import matrix_opt


from local_controllers import DiscreteLQRController, DiscreteModelBasedController

class LinearModel():
    """
    Utility class representing a linear dynamics model. Used internally by EMRLAgent.
    """
    
    def __init__(self, state_dim, action_dim, limits_low=None, limits_high=None):
        """
        Initialize linear model given state, control dimension.

        Args:
            state_dim: Number of state dimensions (assumed real vector space).
            action_dim: Number of action dimensions (assumed real vector space).
            limits_low: State lower limits.
            limts_high: State upper limits.

        Returns:
            None
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.limits_low = limits_low
        self.limits_high = limits_high
        
        self.A = np.random.randn(state_dim, state_dim)
        while np.linalg.det(self.A) == 0:
            print("Re-generating A")
            self.A = np.random.randn(state_dim, state_dim)
        self.B = np.random.randn(state_dim, action_dim)
        self.Q = np.eye(state_dim)
        self.R = np.eye(action_dim)

    def fit(self, data):
        """
        Fit this linear model to the input data, which should be (state,
        action, reward, state_prime, done) tuples.
        
        Args:
            data: The training data, format TBD. TODO

        Returns:
            None
        """
        states = [np.array(trans[0]) for trans in data]
        actions = [np.array(trans[1]) for trans in data]
        state_primes = [np.array(trans[3])for trans in data]

        X = np.stack(states).T
        U = np.stack(actions).T
        Y = np.stack(state_primes).T

        # TODO : Process rewards in data to costs using cost = max_reward - reward
        #        and fit quadratic model of cost function.

        A, B = matrix_opt.fit_matrix_sum_system(X, U, Y)
        #print("A = {}, B = {}".format(A, B))
        self.A = A
        self.B = B

    def predict_state(self, state, action):
        """
        Predict the next state according to the current linear model, given the
        current state and action taken.

        Args:
            state: The current state. Must have dimensionality equal to state_dim.
            action: The current action. Must have dimensionality equal to action_dim.

        Returns:
            The predicted next state.
        """
        assert(len(state) == self.state_dim)
        assert(len(action) == self.action_dim)

        if self.limits_low is not None and self.limits_high is not None:
            # TODO : Clean up
            real_state = np.maximum(self.limits_low, np.minimum(self.limits_high, np.dot(self.A, np.array(state)) + np.dot(self.B, np.array(action))))
            return real_state 
        else:
            return np.dot(self.A, np.array(state)) + np.dot(self.B, np.array(action))

    def predict_cost(self, state, action):
        """
        Predict the cost for the input state and action according to the
        current quadratic model.

        Args:
            state: The current state. Must have dimensionality equal to state_dim.
            action: The current action. Must have dimensionality equal to action_dim.

        Returns:
            The predicted cost.
        """
        assert(len(state) == self.state_dim)
        assert(len(action) == self.action_dim)

        return np.dot(np.array(state).transpose(), np.dot(self.Q, np.array(state))) + np.dot(np.array(action).transpose(), np.dot(self.R, np.array(action)))

    def create_lqr_controller(self, target_point):
        """
        Solve LQR system for the current linear model and deviation from the
        input other_point to obtain a controller which drives the modeled
        system to the target point.

        Args:
            target_point: The target point to control toward.

        Returns:
            An optimal controller for the modeled LQR problem.
        """
        # TODO : Think about whether there's a need for an additional control term directly
        #        encoding movement toward target point.
        return DiscreteLQRController(target_point, self.A, self.B, self.Q, self.R)

    def create_model_based_controller(self, target_point):
        """
        Create a model-based controller which approximately optimizes
        single-step movement toward a target point.

        Args:
            target_point: The target point to control toward.

        Returns:
            A model-based local controller for the model problem.
        """
        # TODO : Think about whether there's a need for an additional control term directly
        #        encoding movement toward target point.
        return DiscreteModelBasedController(target_point, self.A, self.B, self.Q, self.R)

