import numpy as np
import matrix_opt
import viz_1_1d

X = np.eye(5) * 2
Y = np.arange(25).reshape((5, 5))
Z = np.eye(5)

# ===========================
# Matrix mult fit testing
# ===========================

# Things to fit
A = matrix_opt.fit_matrix_system(X, Y)
print("AX = ")
print(np.dot(A, X))
print("\n")

# ===========================
# Matrix sum fit testing
# ===========================
A, B = matrix_opt.fit_matrix_sum_system(X, Y, Z, debug=True)
print("AX + BY = ")
print(np.dot(A, X) + np.dot(B, Y))
print("Computed norm: ")
print(np.linalg.norm(Z - (np.dot(A, X) + np.dot(B, Y))))
print("\n")


# ===========================
# Quadratic form fit testing
# ===========================

Q = np.eye(5)
Q[0,1] = 1
Q[1,0] = 1

state_list = []
target_val_list = []

# Generate points randomly
# TODO : Generate using basis?
num_points = 10
for i in range(num_points):
    x = np.random.randn(5, 1)
    state_list.append(x)
    target_val_list.append(np.dot(x.transpose(), np.dot(Q, x))[0][0])

Q = matrix_opt.fit_quadratic_form_pos_def(state_list, target_val_list)
print("Q from random points = ")
print(Q)
print("Rounded Q from random points = ")
print(np.round(Q))
#for i in range(num_points):
#    print("Estimated form evaluation for state {}".format(i))
#    print(np.dot(state_list[i].transpose(), np.dot(Q, state_list[i])))
#
#    print("Actual value was:")
#    print(target_val_list[i])
#    print("=======\n\n")

print("\nUsing basis:")
for i in range(5):
    x = np.zeros((5, 1))
    x[i] = 1
    state_list.append(x)
    target_val_list.append(np.dot(x.transpose(), np.dot(Q, x))[0][0])

Q = matrix_opt.fit_quadratic_form_pos_def(state_list, target_val_list)
print("\tEstimated Q = ")
print(Q)
print("\tRounded Q = ")
print(np.round(Q))

# =======================
# Control fit testing
# =======================
A = np.eye(5)
B = np.eye(5)
s = np.zeros((5, 1))
s[2] = -1.0
target = np.ones((5, 1)) * 2
target[3] = 0.0

u = matrix_opt.fit_control(target, s, A, B, debug=True)
print("Got estimated u = {}".format(u))
print("Resulting movement As + Bu = {}".format(np.dot(A, s) + np.dot(B, u)))
