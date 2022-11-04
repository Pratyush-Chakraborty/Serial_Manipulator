import sympy as sp
from sympy import *

# assigning symbols to all our variables
theta1, theta2, theta3, theta, a, d, alpha, alpha1, alpha2, alpha3, x, y, z, l1, l2, l3 = sp.symbols("theta1 theta2 theta3 "
                                                                                         "theta a d alpha alpha1"
                                                                                         " alpha2 alpha3 x y z l1 l2 l3")

# creating the rotation matrix of the link about z-axis
rot_z = sp.Matrix(4, 4, [cos(theta), -sin(theta), 0, 0, sin(theta), cos(theta), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

# creating the translational matrix of the link offset
link_offset = sp.Matrix(4, 4, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, d, 0, 0, 0, 1])

# creating the rotation matrix of the link about x-axis
rot_x = sp.Matrix(4, 4, [1, 0, 0, 0, 0, cos(alpha), -sin(alpha), 0, 0, sin(alpha), cos(alpha), 0, 0, 0, 0, 1])

# printing the length of the link matrix
link_length = sp.Matrix(4, 4, [1, 0, 0, a, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

# creating the homogenous transformation matrix
homogenous_trans_matrix = rot_z*link_length*rot_x*link_offset

# printing the homogenous transformation matrix
print("printing the homogenous transformation matrix")
sp.pretty_print(homogenous_trans_matrix)

print("--------------------------------------------------------------------")

# homogenous matrix when moving from frame 0 to 3
homomat_0to1 = homogenous_trans_matrix.subs([(theta, theta1), (a, l1), (d, 0), (alpha, 0)])
homomat_1to2 = homogenous_trans_matrix.subs([(theta, theta2), (a, l2), (d, 0), (alpha, 0)])
homomat_2to3 = homogenous_trans_matrix.subs([(theta, theta3), (a, l3), (d, 0), (alpha, 0)])

# final homogenous matrix of the serial manipulator
final_homo_mat = homomat_0to1*homomat_1to2*homomat_2to3

# printing the final homogenous matrix of the serial manipulator
print("printing the final homogenous matrix of the serial manipulator")
sp.pretty_print(final_homo_mat)

print("--------------------------------------------------------------------")

# simplifying the final homogenous matrix of the serial manipulator
simplified_trans_mat = trigsimp(final_homo_mat)

# printing the simplified version of the transformation matrix of the serial manipulator
print("printing the simplified version of the transformation matrix of the serial manipulator")
sp.pretty_print(simplified_trans_mat)

