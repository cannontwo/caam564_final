import scipy
import scipy.linalg
import numpy as np
import control
import matrix_opt

class DiscreteLQRController:
    """
    Class representing a discrete LQR optimal controller. Mostly used to
    abstract the interior library calls for solving the Ricatti equation and
    transform things to a nice interface. It is assumed that this controller
    will try to drive the input system to a target point, so that all input
    states will be controlled as (target_point - state).
    """

    def __init__(self, target_point, A, B, Q, R, limits_low=None, limits_high=None):
        """
        Initialize the Discrete LQR Controller with a system description, then
        solve for the optimal feedback controller.

        Args:
            target_point: The target point that the system will be controlled
                          to.
            A: The state transition matrix.
            B: The control transition matrix.
            Q: The state cost matrix.
            R: The control cost matrix.
            limits_low: Lower action limits.
            limits_high: Upper action limits.

        Returns:
            None
        """
        self.target_point = np.array(target_point)
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)

        self.limits_low = limits_low
        self.limits_high = limits_high

        rescaled = False
        if np.linalg.norm(self.B) < 1e-6:
            self.B = np.multiply(self.B, 1e6)
            rescaled = True

        # Find solution to discrete-time Algebraic Ricatti Equation
        try:
            #self.X = np.array(scipy.linalg.solve_discrete_are(self.A, self.B, 
            #      self.Q, self.R))

            # Find optimal control gains
            #self.K = np.array(scipy.linalg.inv(np.dot(np.dot(np.dot(B.T,self.X),B) +
            #    R),np.dot(B.T, np.dot(self.X, A))))

            self.X, _, self.K = control.dare(self.A, self.B, self.Q, self.R)
            #print("Solved Ricatti equation for A={}, B={}, B_norm={}".format(self.A, self.B, np.linalg.norm(self.B)))
        except:
            print("Couldn't solve Ricatti equation for A={}, B={}, B_norm={}".format(self.A, self.B, np.linalg.norm(self.B)))
            self.K = np.random.randn(self.B.shape[1], self.A.shape[0])

    def get_optimal_action(self, state):
        """
        Get the optimal action for the input state in order to move toward the
        previously configured target point.

        Args:
            state: The current state of the system.

        Returns:
            The optimal action to take.
        """
        
        if self.limits_low != None and self.limits_high != None:
            real_control = -np.maximum(self.limits_low, np.minimum(self.limits_high, np.dot(self.K, self.target_point - np.array(state))))

            return real_control        
        else:
            return -np.dot(self.K, self.target_point - np.array(state))

class DiscreteModelBasedController:
    """
    Class used to represent a controller that just tries to solve a linear
    system to a target point in terms of the control input. Ignores Q and R for
    now.
    """

    def __init__(self, target_point, A, B, Q, R, limits_low=None, limits_high=None):
        """
        Initialize the Discrete LQR Controller with a system description, then
        solve for the optimal feedback controller.

        Args:
            target_point: The target point that the system will be controlled
                          to.
            A: The state transition matrix.
            B: The control transition matrix.
            Q: The state cost matrix.
            R: The control cost matrix.
            limits_low: Lower action limits.
            limits_high: Upper action limits.

        Returns:
            None
        """
        self.target_point = np.array(target_point)
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)

        self.limits_low = limits_low
        self.limits_high = limits_high

    def get_optimal_action(self, state):
        """
        Get the optimal action for the input state in order to move toward the
        previously configured target point.

        Args:
            state: The current state of the system.

        Returns:
            The optimal action to take.
        """
        # TODO : Implement solving linear system, replace tmp with correct control
        control = matrix_opt.fit_control(self.target_point, state, self.A, self.B)
        if self.limits_low != None and self.limits_high != None:
            real_control = -np.maximum(self.limits_low, np.minimum(self.limits_high, control))

            return real_control        
        else:
            return control
