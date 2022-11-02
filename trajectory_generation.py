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

# asking the user what they wanted to trace
object = input("Do you want to bring the arm to a specific point or trace a straight line:- ")

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

# creating the inverse of the jacobian matrix
jacobian_inverse = jacobian_matrix.inv()

# creating the initial guess matrix
guess_matrix = sp.Matrix(3, 1, [init_theta1, init_theta2, init_theta3])

# creating the value matrix list
value_matrix = sp.Matrix(3, 1, [0, 0, 0])

# part for straight line
if object == "straight line":
    # drawing a straight line
    x_in = int(input("Tell the x coordinate of the starting point of the line:- "))
    y_in = int(input("Tell the y coordinate of the starting point of the line:- "))
    phi_in = float(input("Tell the starting orientation of the line:- "))
    x_fi = int(input("Tell the x coordinate of the starting point of the line:- "))
    y_fi = int(input("Tell the x coordinate of the starting point of the line:- "))
    phi_fi = float(input("Tell the x coordinate of the starting point of the line:- "))
    # x_in, y_in, phi_in = 5, 5, 0.95
    # x_fi, y_fi, phi_fi = 8, 5, 0.75
    x_matrix = np.linspace(x_in, x_fi, num=int(50*(abs(x_in-x_fi))))
    y_matrix = np.linspace(y_in, y_fi, num=int(50*(abs(y_in-y_fi))))
    phi_matrix = np.linspace(phi_in, phi_fi, num=int(50*(abs(phi_in-phi_fi))))
    x_matrix_len = len(x_matrix)
    y_matrix_len = len(y_matrix)
    phi_matrix_len = len(phi_matrix)

    # finding the loop rate for the number of iterations
    loop_rate = max(x_matrix_len, y_matrix_len, phi_matrix_len)

    # animating the serial manipulator
    animate_theta1 = [0]
    animate_theta2 = [0]
    animate_theta3 = [0]

    # using a for loop for finding all the angles of link for all the equally divided points between the initial and final point
    for g in range (0, loop_rate):
        i = 0
        x1_eqn = l1 * cos(theta1) + l2 * cos(theta1 + theta2) + l3 * cos(theta1 + theta2 + theta3)
        y1_eqn = l1 * sin(theta1) + l2 * sin(theta1 + theta2) + l3 * sin(theta1 + theta2 + theta3)
        phi1_eqn = cos(theta1 + theta2 + theta3)

        if g < x_matrix_len:
            x1_eqn = x1_eqn - x_matrix[g]
        else:
            x1_eqn = x1_eqn - x_fi
        if g < y_matrix_len:
            y1_eqn = y1_eqn - y_matrix[g]
        else:
            y1_eqn = y1_eqn - y_fi
        if g < phi_matrix_len:
            phi1_eqn = phi1_eqn - phi_matrix[g]
        else:
            phi1_eqn = phi1_eqn - cos(phi_fi)


        # Doing the calculation by Newton Raphson method
        # creating the loop for iteration till the approximated value is reached
        while i == 0:
            # matrix of the function after value substitution
            function_value1 = x1_eqn.subs(
                [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

            function_value2 = y1_eqn.subs(
                [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

            function_value3 = phi1_eqn.subs(
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

        # printing the reduced value of the angles in the 0 to 2*pi range
        animate_theta1.append(negative_red_value_mat[0,0])
        animate_theta2.append(negative_red_value_mat[1,0])
        animate_theta3.append(negative_red_value_mat[2,0])

    # creating the necessary reference line and graph
    figure = plt.figure()
    ax = plt.axes(xlim=(-1, 15), ylim=(-5, 15))
    link1, = ax.plot([], [], lw=2, marker='o')

    # calculating the number of frames
    no_of_frames = loop_rate

    # creating the call back function for animation
    def anim(i):
        x1, y1 = 0, 0
        angle1, angle2, angle3 = animate_theta1[i], animate_theta2[i], animate_theta3[i]
        x2 = l1 * cos(angle1)
        y2 = l1 * sin(angle1)
        x3 = x2 + l2 * cos(angle1 + angle2)
        y3 = y2 + l2 * sin(angle1 + angle2)
        x4 = x3 + l3 * cos(angle1 + angle2 + angle3)
        y4 = y3 + l3 * sin(angle1 + angle2 + angle3)

        # updating the link object for plotting in the current frame
        link1.set_data([x1, x2, x3, x4], [y1, y2, y3, y4])
        plt.scatter(x4, y4, marker = '.', lw = 0.15, color = 'red')

    # writing the FuncAnimation code 
    animate = animation.FuncAnimation(figure, anim, frames=no_of_frames, repeat=False, interval=5)

    # showing the animation
    plt.show()

    writergif = animation.PillowWriter(fps=10) 
    animate.save('tracing.gif', writer=writergif)

# part for going to a simple point
if object == 'specific point':
    # creating the initial guess matrix
    guess_matrix = sp.Matrix(3, 1, [init_theta1, init_theta2, init_theta3])

    # creating the value matrix list
    value_matrix = sp.Matrix(3, 1, [0, 0, 0])

    # Doing the calculation by Newton Raphson method
    # creating the loop for iteration till the approximated value is reached
    while i == 0:
        # matrix of the function after value substitution
        function_value1 = x1_eqn.subs(
            [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

        function_value2 = y1_eqn.subs(
            [(theta1, guess_matrix[0, 0]), (theta2, guess_matrix[1, 0]), (theta3, guess_matrix[2, 0])])

        function_value3 = phi1_eqn.subs(
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

    # Now printing the final results
    print("--------------The final roots of the functions-----------------")
    sp.pretty_print(value_matrix)

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

    # printing the reduced value of the angles in the 0 to 2*pi range
    print("-----------------The final roots in the 0 to 2*pi range-----------------------")
    sp.pretty_print(negative_red_value_mat)

    # plotting one of the position of the link
    # link 1 position
    x1_coordinate = l1*cos(negative_red_value_mat[0, 0])
    y1_coordinate = l1*sin(negative_red_value_mat[0, 0])
    # link 2 position
    x2_coordinate = x1_coordinate + l2*cos(negative_red_value_mat[1, 0] + negative_red_value_mat[0, 0])
    y2_coordinate = y1_coordinate + l2*sin(negative_red_value_mat[1, 0] + negative_red_value_mat[0, 0])

    # plotting the graph
    plt.plot([0, x1_coordinate], [0, y1_coordinate], label='link 1')
    plt.scatter([0, x1_coordinate], [0, y1_coordinate])
    plt.plot([x1_coordinate, x2_coordinate], [y1_coordinate, y2_coordinate], label='link 2')
    plt.scatter([x1_coordinate, x2_coordinate], [y1_coordinate, y2_coordinate])
    plt.plot([x2_coordinate, final_x], [y2_coordinate, final_y], label = "link 3")
    plt.scatter([x2_coordinate, final_x], [y2_coordinate, final_y])
    plt.legend()

    # showing the graph
    plt.show()

    # animating the serial manipulator
    animate_theta1 = [0]
    animate_theta2 = [0]
    animate_theta3 = [0]

    # creating the necessary reference line and graph
    figure = plt.figure()
    ax = plt.axes(xlim=(-5, 15), ylim=(-5, 15))
    link1, = ax.plot([], [], lw=2, marker='o')

    # deciding the number of frames
    l = 0
    no_of_frames = max(abs(negative_red_value_mat[0, 0] // float((0.3 * pi) / 180)), abs(negative_red_value_mat[1, 0] // float((0.3 * pi) / 180)), abs(negative_red_value_mat[2, 0] // float((0.3 * pi) / 180)))
    print(no_of_frames)
    for l in range(1, no_of_frames + 1):
        if abs(animate_theta1[l-1]) < (negative_red_value_mat[0, 0]):
            animate_theta1.append(l * float((0.3 * pi) / 180))
        elif abs(animate_theta1[l-1]) < (-1) * negative_red_value_mat[0, 0]:
            animate_theta1.append((-1) * l * float((0.3 * pi) / 180))
        else:
            animate_theta1.append(negative_red_value_mat[0, 0])
        if abs(animate_theta2[l-1]) < (negative_red_value_mat[1, 0]):
            animate_theta2.append(l * float((0.3 * pi) / 180))
        elif abs(animate_theta2[l-1]) < (-1) * negative_red_value_mat[1, 0]:
            animate_theta2.append((-1) * l * float((0.3 * pi) / 180))
        else:
            animate_theta2.append(negative_red_value_mat[1, 0])
        if abs(animate_theta3[l-1]) < (negative_red_value_mat[2, 0]):
            animate_theta3.append(l * float((0.3 * pi) / 180))
        elif abs(animate_theta3[l-1]) < (-1) * negative_red_value_mat[2, 0]:
            animate_theta3.append((-1) * l * float((0.3 * pi) / 180))
        else:
            animate_theta3.append(negative_red_value_mat[2, 0])
    print("/n")
    print(animate_theta1)
    print("/n")
    print(animate_theta2)
    print("/n")
    print(animate_theta3)
    end_effec_pos = []
    # creating the call back function for animation
    def anim(i):
        x1, y1 = 0, 0
        angle1, angle2, angle3 = animate_theta1[i], animate_theta2[i], animate_theta3[i]
        x2 = l1 * cos(angle1)
        y2 = l1 * sin(angle1)
        x3 = x2 + l2 * cos(angle1 + angle2)
        y3 = y2 + l2 * sin(angle1 + angle2)
        x4 = x3 + l3 * cos(angle1 + angle2 + angle3)
        y4 = y3 + l3 * sin(angle1 + angle2 + angle3)
        
        # updating the link object for plotting in the current frame
        link1.set_data([x1, x2, x3, x4], [y1, y2, y3, y4])
        plt.scatter(x4, y4, marker = '.', lw = 0.15, color = 'red')

    # writing the FuncAnimation code 
    animate = animation.FuncAnimation(figure, anim, frames=no_of_frames, repeat=False, interval=20)

    # showing the animation
    plt.show()

writergif = animation.PillowWriter(fps=10) 
animate.save('tracing.gif', writer=writergif)