import math
import numpy as np
from pyomo.environ import *

# Location of the solver that pyomo uses on the backend
ipopt_executable = '/home/cannon/Documents/pyomo_testing/ipopt'

def matrix_mult(n, m, l, A, X): 
    """
    Helper function to multiply the matrices A and X, where
    A is assumed to be represented as a pyomo indexed variable.

    Args:
        n: The number of rows in A.
        m: The number of columns in A and number of rows in X.
        l: The number of columns in X.
        A: The matrix being optimized, represented as a pyomo variable.
        X: A numpy matrix of appropriate size.

    Returns:
        The matrix AX.
    """
    ret_mat = [[] for _ in range(n)]

    for i in range(n):
        for j in range(l):
            ret_mat[i].append(0)
            for k in range(m):
                ret_mat[i][j] += A[i,k] * X[k,j]

    return ret_mat

def matrix_add(n, m, A, B):
    """
    Helper function to add the matrices A and B, probably represented as pyomo variables.
    
    Args:
        n: The number of rows in A and B.
        m: The number of columns in A and B.
        A: The first matrix to add.
        B: The second matrix to add.
        
    Returns:
        The matrix A + B.
    """
    ret_mat = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            ret_mat[i].append(A[i][j] + B[i][j])

    return ret_mat

def matrix_norm(A, n, m):
    """
    Helper function to compute the Frobenius norm squared of the 
    matrix A. Necessary so that this can
    serve as a pyomo optimization function.
    Args:
        A: The matrix.
        n: The number of rows in A.
        m: The number of columns in A.

    Returns:
        The Frobenius norm of the matrix.
    """
    norm = 0.0
    for i in range(n):
        for j in range(m):
            norm += (A[i,j])**2

    return norm

def matrix_diff_norm(Z, Y, n, m):
    """
    Helper function to compute the Frobenius norm squared of the 
    difference between two matrices. Necessary so that this can
    serve as a pyomo optimization function.
Args:
        Z: The first matrix.
        Y: The second matrix.
        n: The number of rows in Z and Y.
        m: The number of columns in Z and Y.

    Returns:
        The Frobenius norm of the element-wise difference between the two
        matrices.
    """
    norm = 0.0
    for i in range(n):
        for j in range(m):
            norm += (Z[i][j] - Y[i][j])**2

    return norm

def matrix_l1_penalty(A, n, m):
    """
    Helper function to compute the L1 norm of a matrix
    Args:
        A: The matrix.
        n: The number of rows in A.
        m: The number of columns in A.

    Returns:
        The L1 norm.
    """
    norm = 0.0
    for i in range(n):
        for j in range(m):
            norm += abs(A[i,j])

    return norm

def make_mat_from_model(n, m, A):
    """
    Helper function to read out a matrix represented as a numpy array from
    a pyomo model variable A.
    
    Args:
        n: The number of rows in the model matrix.
        m: The number of columns in the model matrix.
        A: The pyomo model variable to read out of.

    Returns:
        The assembled matrix.
    """
    mat_list = [round(A[i,j]()) for i in range(n) for j in range(m)]
    return np.array(mat_list).reshape((n,m))

def noeval_make_mat_from_list(n, m, A):
    """
    Helper function to read out a matrix represented as a numpy array from
    a list of lists A, without evaluating the model variables.
    
    Args:
        n: The number of rows in the model matrix.
        m: The number of columns in the model matrix.
        A: The pyomo model variable to read out of.

    Returns:
        The assembled matrix.
    """
    mat_list = [A[i][j] for i in range(n) for j in range(m)]
    return np.array(mat_list).reshape((n,m))

def make_mat_from_model_exact(n, m, A):
    """
    Helper function to read out a matrix represented as a numpy array from
    a pyomo model variable A.
    
    Args:
        n: The number of rows in the model matrix.
        m: The number of columns in the model matrix.
        A: The pyomo model variable to read out of.

    Returns:
        The assembled matrix.
    """
    mat_list = [A[i,j]() for i in range(n) for j in range(m)]
    return np.array(mat_list).reshape((n,m))

def make_lower_triangular_mat_from_model(n, A):
    """
    Helper function to assemble a lower triangular matrix represented by 
    a pyomo variable into a numpy array. A is assumed to be square.
    
    Args:
        n: The number of rows, columns in the model matrix.
        A: The pyomo model variable to read out of.

    Returns:
        The assembled matrix.
    """
    mat_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j <= i:
                mat_list[i].append(A[i,j]())
            else:
                mat_list[i].append(0.0)

    return np.array(mat_list)

def make_lower_triangular_mat_from_model_no_eval(n, A):
    """
    Helper function to assemble a lower triangular matrix represented by 
    a pyomo variable into a numpy array. A is assumed to be square.
    
    Args:
        n: The number of rows, columns in the model matrix.
        A: The pyomo model variable to read out of.

    Returns:
        The assembled matrix.
    """
    mat_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j <= i:
                mat_list[i].append(A[i,j])
            else:
                mat_list[i].append(0.0)

    return np.array(mat_list)

def fit_matrix_system(X, Y, debug=False):
    """
    Function to solve a matrix multiplication problem of the form
    AX = Y using optimization and the squared Frobenius norm as 
    objective function.

    Args:
        X: The matrix to be left-multiplied by A.
        Y: The matrix that should be more-or-less equal to AX.
    
    Returns:
        A, the fit square matrix which solves the system.
    """
    assert(len(X.shape) == 2)
    assert(len(Y.shape) == 2)
    assert(Y.shape[0] == X.shape[0])
    assert(Y.shape[1] == X.shape[1])

    n, m = X.shape

    # Create a model instance
    model = ConcreteModel()

    model.i = Set(initialize=[i for i in range(n)])
    model.j = Set(initialize=[j for j in range(n)])

    # Create the decision variable
    # TODO : Might want to allow passing in bounds
    model.A = Var(model.i, model.j, domain=Reals, initialize=1.0, bounds=(-10.0, 10.0))

    # Define objective
    Z = matrix_mult(n, n, m, model.A, X)
    local_obj = lambda model : matrix_diff_norm(Z,Y,n,m)

    # Create the objective
    model.norm = Objective(rule=local_obj, sense=minimize)

    
    if (debug):
        print("A at start = ")
        print(make_mat_from_model(n, n, model.A))

    # solve using the nonlinear solver ipopt
    SolverFactory('ipopt', executable=ipopt_executable).solve(model)

    if (debug):
        # print solution
        print('Exact A at solution = ')
        print(make_mat_from_model_exact(n, n, model.A))
        print('Approximate A at solution = ')
        print(make_mat_from_model(n, n, model.A))
        print('Min diff norm =', model.norm())

    return make_mat_from_model_exact(n, n, model.A)

def fit_matrix_sum_system(X, Y, Z, debug=False):
    """
    Function to solve an affine matrix equation of the form
    AX + BY = Z using optimization and the squared Frobenius norm
    as objective function.

    Args:
        X: The matrix to be left-multiplied by A.
        Y: The matrix to be left-multiplied by B.
        Z: The matrix that should be more-or-less equal to AX + BY.
    
    Returns:
        (A, B), the fit square matrices which solve the system.

    """
    assert(len(X.shape) == 2)
    assert(len(Y.shape) == 2)
    assert(len(Z.shape) == 2)
    assert(Y.shape[1] == X.shape[1])
    assert(Z.shape[1] == X.shape[1])

    n, m = X.shape
    l, _ = Y.shape
    p, _ = Z.shape

    # Create a model instance
    model = ConcreteModel()

    model.i = Set(initialize=[i for i in range(n)])
    model.j = Set(initialize=[j for j in range(n)])
    model.k = Set(initialize=[k for k in range(l)])
    model.p = Set(initialize=[p for p in range(p)])

    # Create the decision variables
    # TODO : Might want to allow passing in bounds
    model.A = Var(model.p, model.j, domain=Reals, initialize=0.0, bounds=(-10.0, 10.0))
    model.B = Var(model.p, model.k, domain=Reals, initialize=0.0, bounds=(-10.0, 10.0))

    # Initialize to identity
    for i in range(min(p, n)):
        model.A[i, i] = 1.0

    for i in range(min(p, l)):
        model.B[i, i] = 1.0

    # Define objective
    C = matrix_add(p, m, matrix_mult(p, n, m, model.A, X), matrix_mult(p, l, m, model.B, Y))
    local_obj = lambda model : matrix_diff_norm(Z,C,p,m)

    # Create the objective
    model.norm = Objective(rule=local_obj, sense=minimize)

    if (debug):
        print("A at start = ")
        print(make_mat_from_model(p, n, model.A))
        print("B at start = ")
        print(make_mat_from_model(p, l, model.B))

    # solve using the nonlinear solver ipopt
    SolverFactory('ipopt', executable=ipopt_executable).solve(model)

    if (debug):
        # print solution
        print('Exact A at solution = ')
        print(make_mat_from_model_exact(p, n, model.A))
        print('Exact B at solution = ')
        print(make_mat_from_model_exact(p, l, model.B))

        print('Approximate A at solution = ')
        print(make_mat_from_model(p, n, model.A))
        print('Approximate B at solution = ')
        print(make_mat_from_model(p, l, model.B))
        print('Min diff norm =', model.norm())

    return (make_mat_from_model_exact(p, n, model.A), make_mat_from_model_exact(p, l, model.B))

def fit_quadratic_form(state_list, target_val_list, debug=False):
    """
    Function to solve a quadratic form equation using a list of 
    sampled data. The equation is assumed to be of the form
    r = x^T Q x, and this function estimates Q given example
    x, r.

    Args:
        state_list: The list of x's.
        target_val_list: The list of r's.

    Returns:
        The estimated Q.
    """
    # All states must have the same dimensions and be vectors.
    initial_state_dim = state_list[0].shape
    assert(len(initial_state_dim) == 2)
    assert(initial_state_dim[1] == 1)
    for state in state_list:
        assert(state.shape == initial_state_dim)
    
    n, _ = initial_state_dim


    # Create model
    model = ConcreteModel()

    model.i = Set(initialize=[i for i in range(n)])
    model.j = Set(initialize=[j for j in range(n)])

    # Create the decision variable
    # TODO : Might want to allow passing in bounds
    model.Q = Var(model.i, model.j, domain=Reals, initialize=1.0, bounds=(-10.0, 10.0))

    # We want to minimize absolute deviation from estimation of the quadratic form.
    total = 0
    for state, target in zip(state_list, target_val_list):
        # Define objective
        Z = matrix_mult(n, n, 1, model.Q, state)
        scalar = matrix_mult(1, n, 1, state.transpose(),
                noeval_make_mat_from_list(n, 1, Z))
        total += (scalar[0][0] - target)**2
    
    local_obj = lambda model : total

    # Create the objective
    model.norm = Objective(rule=local_obj, sense=minimize)

    if (debug):
        print("Q at start = ")
        print(make_mat_from_model(n, n, model.Q))

    # solve using the nonlinear solver ipopt
    SolverFactory('ipopt', executable=ipopt_executable).solve(model)

    if (debug):
        # print solution
        print('Exact Q at solution = ')
        print(make_mat_from_model_exact(n, n, model.Q))
        print('Approximate Q at solution = ')
        print(make_mat_from_model(n, n, model.Q))
        print('Min diff norm =', model.norm())

        avg = math.sqrt(model.norm()) / float(len(state_list))
        print('Average square root of norm = {}'.format(avg))

    return make_mat_from_model_exact(n, n, model.Q)

def fit_quadratic_form_pos_def(state_list, target_val_list, debug=False):
    """
    Function to solve a quadratic form equation using a list of 
    sampled data. The equation is assumed to be of the form
    r = x^T Q x, and this function estimates Q given example
    x, r. Q is assumed to be positive semidefinite, and this is 
    enforced by optimizing only the lower-triangular factor L
    s.t. Q = L*L^t. 

    Args:
        state_list: The list of x's.
        target_val_list: The list of r's.

    Returns:
        The estimated Q.
    """
    # All states must have the same dimensions and be vectors.
    initial_state_dim = state_list[0].shape
    assert(len(initial_state_dim) == 2)
    assert(initial_state_dim[1] == 1)
    for state in state_list:
        assert(state.shape == initial_state_dim)
    
    n, _ = initial_state_dim


    # Create model
    model = ConcreteModel()

    model.i = Set(initialize=[i for i in range(n)])
    model.j = Set(initialize=[j for j in range(n)])

    # Create the decision variable
    # TODO : Might want to allow passing in bounds
    model.L = Var(model.i, model.j, domain=Reals, initialize=1.0, bounds=(-10.0, 10.0))

    # We want to minimize absolute deviation from estimation of the quadratic form.
    total = 0
    for state, target in zip(state_list, target_val_list):
        # Define objective
        mat_L = make_lower_triangular_mat_from_model_no_eval(n, model.L)
        Q = np.dot(mat_L, mat_L.transpose())
        Z = matrix_mult(n, n, 1, Q, state)
        scalar = matrix_mult(1, n, 1, state.transpose(),
                noeval_make_mat_from_list(n, 1, Z))
        total += (scalar[0][0] - target)**2
    
    local_obj = lambda model : total

    # Create the objective
    model.norm = Objective(rule=local_obj, sense=minimize)

    if (debug):
        print("Q at start = ")
        L = make_lower_triangular_mat_from_model(n, model.L)
        print(np.dot(L, L.transpose()))

    # solve using the nonlinear solver ipopt
    SolverFactory('ipopt', executable=ipopt_executable).solve(model)

    L = make_lower_triangular_mat_from_model(n, model.L)
    if (debug):
        # print solution
        print('Exact Q at solution = ')
        print(np.dot(L, L.transpose()))
        print('Min diff norm =', model.norm())

        avg = math.sqrt(model.norm()) / float(len(state_list))
        print('Average square root of norm = {}'.format(avg))

    return np.dot(L, L.transpose())

def fit_quadratic_form_sum_pos_def(state_list, control_list, target_val_list, debug=False):
    """
    Function to solve a quadratic form sum equation using a list of 
    sampled data. The equation is assumed to be of the form
    r = x^T Q x + y^t R y, and this function estimates Q, R given example
    x, y, r. Q, R are assumed to be positive semidefinite, and this is 
    enforced by optimizing only the lower-triangular factors L, M
    s.t. Q = L*L^t, R = M*M^t. 

    Args:
        state_list: The list of x's.
        control_list: The list of y's.
        target_val_list: The list of r's.

    Returns:
        The estimated (Q, R).
    """
    # TODO : Implement
    pass

def fit_control(target_state, state, A, B, debug=False):
    """
    Function to solve a linear system for the control which moves the system
    closest to the desired target point. In other words, solves a system of the
    form s' = As + Bu for the u which minimizes ||s' - (As + Bu)S||. No
    restrictions are assumed on A or B.
    """
    # TODO : Finish implementing
    assert(len(A.shape) == 2)
    assert(len(B.shape) == 2)
    assert(len(target_state.shape) == 2)
    assert(len(state.shape) == 2)

    assert(A.shape[0] == A.shape[1])
    assert(B.shape[0] == A.shape[0])
    assert(target_state.shape[1] == 1)
    assert(target_state.shape[0] == A.shape[0])
    assert(state.shape[1] == 1)
    assert(state.shape[0] == A.shape[0])

    n, _ = A.shape
    _, m = B.shape

    # Create a model instance
    model = ConcreteModel()

    model.i = Set(initialize=[i for i in range(m)])
    model.j = Set(initialize=[0])

    # Create the decision variables
    # TODO : Might want to allow passing in bounds
    model.u = Var(model.i, model.j, domain=Reals, initialize=1.0, bounds=(-10.0, 10.0))

    # Define objective
    C = matrix_add(n, 1, matrix_mult(n, n, 1, A, state), matrix_mult(n, m, 1, B, model.u))
    local_obj = lambda model : matrix_diff_norm(target_state,C,n,1)

    # Create the objective
    model.norm = Objective(rule=local_obj, sense=minimize)

    
    if (debug):
        print("u at start = ")
        print(make_mat_from_model(m, 1, model.u))

    # solve using the nonlinear solver ipopt
    SolverFactory('ipopt', executable=ipopt_executable).solve(model)

    if (debug):
        # print solution
        print('Exact u at solution = ')
        print(make_mat_from_model_exact(m, 1, model.u))

        print('Approximate u at solution = ')
        print(make_mat_from_model(m, 1, model.u))
        print('Min diff norm =', model.norm())

    return make_mat_from_model_exact(m, 1, model.u)

