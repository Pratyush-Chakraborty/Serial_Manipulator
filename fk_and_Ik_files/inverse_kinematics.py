import sympy as sp
import numpy as np
from sympy import cos, sin, pi
from matplotlib import pyplot as plt
from matplotlib import animation

# declare the variables as symbols
theta1, theta2, theta3 = sp.symbols("theta1 theta2 theta")

# taking the input for the link lengths
l1 = 4
l2 = 6
l3 = 5

# taking the input for our end effector final position
final_x = float(input("Enter the final x coordinate position of the end effector :- "))
final_y = float(input("Enter the final y coordinate position of the end effector :- "))
orientation = float(input("Enter the final orientation of the end effector :- "))

# creating the equation for verifying
x_eqn = l1 * cos(theta1) + l2 * cos(theta1 + theta2) + l3 * cos(theta1 + theta2 + theta3) - final_x
y_eqn = l1 * sin(theta1) + l2 * sin(theta1 + theta2) + l3 * sin(theta1 + theta2 + theta3) - final_y
phi = cos(theta1 + theta2 + theta3) - cos(orientation)

# making the initial guesses
init_theta1 = 0.1
init_theta2 = 0.2
init_theta3 = 0.3

# printing the equations
print("-----------------------The equations are-------------------------")
sp.pretty_print(x_eqn)
sp.pretty_print(y_eqn)
sp.pretty_print(phi)

# creating the Jacobian matrix
jacobian = []
# creating the differentiated equations and storing them in a object element of the jacobian matrix
jacobian.append(sp.diff(x_eqn, theta1))
jacobian.append(sp.diff(x_eqn, theta2))
jacobian.append(sp.diff(x_eqn, theta3))
jacobian.append(sp.diff(y_eqn, theta1))
jacobian.append(sp.diff(y_eqn, theta2))
jacobian.append(sp.diff(y_eqn, theta3))
jacobian.append(sp.diff(phi, theta1))
jacobian.append(sp.diff(phi, theta2))
jacobian.append(sp.diff(phi, theta3))

# the appended jacobian matrix
jacobian_matrix = sp.Matrix(3, 3, jacobian)

# printing the jacobian matrix
# print('-------------------------------The jacobian matrix-------------------------')
# sp.pretty_print(jacobian_matrix)

# creating the inverse of the jacobian matrix
jacobian_inverse = jacobian_matrix.inv()

# printing the inverse of the jacobian matrix
# print('----------------------The inverse of the jacobian matrix----------------------')
# sp.pretty_print(jacobian_inverse)

# creating the initial guess matrix
guess_matrix = sp.Matrix(3, 1, [init_theta1, init_theta2, init_theta3])

# creating the value matrix list
value_matrix = sp.Matrix(3, 1, [0, 0, 0])

i = 0

# running a while loop for the approximation using newton raphson method
while i == 0:
    # matrix of the function after value substitution
    function_value1 = x_eqn.subs(
        [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

    function_value2 = y_eqn.subs(
        [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

    function_value3 = phi.subs(
        [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

    function_matrix = sp.Matrix(3, 1, [function_value1, function_value2, function_value3])

    jacobian_inv_value_matrix = jacobian_inverse.subs(
        [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

    # writing the approximation equation
    value_matrix = guess_matrix - jacobian_inv_value_matrix * function_matrix

    # creating an if loop for terminating the iteration
    if guess_matrix[0, 0] - value_matrix[0, 0] < 0.000000001 and guess_matrix[1, 0] - value_matrix[
        1, 0] < 0.000000001 and \
            guess_matrix[2, 0] - value_matrix[2, 0] < 0.000000001:
        i = 1

    # now changing the initial guess values
    guess_matrix = value_matrix
    guess_matrix[0, 0] = value_matrix[0, 0]
    guess_matrix[1, 0] = value_matrix[1, 0]
    guess_matrix[2, 0] = value_matrix[2, 0]

# creating list for the reduced values of angle
red_value_mat = sp.Matrix(3, 1, [value_matrix[0, 0], value_matrix[1, 0], value_matrix[2, 0]])

j = 0
# reducing the value of theta1, theta2, and theta3 to bring in the range of below 360 degrees
while j == 0:
    if red_value_mat[0, 0] > float(pi) or red_value_mat[1, 0] > float(pi) or red_value_mat[2, 0] > float(pi):
        if red_value_mat[0, 0] > float(pi):
            red_value_mat[0, 0] = red_value_mat[0, 0] - float(2 * pi)
        else:
            red_value_mat[0, 0] = red_value_mat[0, 0]
        if red_value_mat[1, 0] > float(pi):
            red_value_mat[1, 0] = red_value_mat[1, 0] - float(2 * pi)
        else:
            red_value_mat[1, 0] = red_value_mat[1, 0]
        if red_value_mat[2, 0] > float(pi):
            red_value_mat[2, 0] = red_value_mat[2, 0] - float(2 * pi)
        else:
            red_value_mat[2, 0] = red_value_mat[2, 0]
    if red_value_mat[0, 0] <= float(pi) and red_value_mat[1, 0] <= float(pi) and red_value_mat[2, 0] <= float(pi):
        j = 1

# creating a list for increasing the value of the negative angles
negative_red_value_mat = sp.Matrix(3, 1, [red_value_mat[0, 0], red_value_mat[1, 0], red_value_mat[2, 0]])

k = 0 
# reducing the value of theta1, theta2, and theta3 to bring in the range of above 0 degrees
while k == 0:
    if negative_red_value_mat[0, 0] < float(-pi) or negative_red_value_mat[1, 0] < float(-pi) or negative_red_value_mat[2, 0] < float(-pi):
        if negative_red_value_mat[0, 0] < float(-pi):
            negative_red_value_mat[0, 0] = negative_red_value_mat[0, 0] + float(2 * pi)
        else:
            negative_red_value_mat[0, 0] = negative_red_value_mat[0, 0]
        if negative_red_value_mat[1, 0] < float(-pi):
            negative_red_value_mat[1, 0] = negative_red_value_mat[1, 0] + float(2 * pi)
        else:
            negative_red_value_mat[1, 0] = negative_red_value_mat[1, 0]
        if negative_red_value_mat[2, 0] < float(-pi):
            negative_red_value_mat[2, 0] = negative_red_value_mat[2, 0] + float(2 * pi)
        else:
            negative_red_value_mat[2, 0] = negative_red_value_mat[2, 0]
    if negative_red_value_mat[0, 0] >= float(-pi) and negative_red_value_mat[1, 0] >= float(-pi) and negative_red_value_mat[2, 0] >= float(-pi):
        k = 1

# printing the calculated angles 
sp.pretty_print(negative_red_value_mat)