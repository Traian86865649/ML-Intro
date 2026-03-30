# imports package that allows me to get vectors
import numpy as np
from scipy.special import softmax

#importing the random package
import random

def get_float_input(prompt):
  while True:
    try:
      user_input = input(prompt)
      float_value = float(user_input)
      return float_value
    except ValueError:
      print("Invalid input. Please enter a valid decimal number.")

# defining ranges for the letters we want the model to identify
a_lb = 0
a_ub = 1.5
b_lb = 1.5
b_ub = 2.5
c_lb = 2.5
c_ub = 3.5
d_lb = 3.5
d_ub = 4.5
e_lb = 4.5
e_ub = 5.5

# getting variables from user to start iteration
variables_input = input("Please input m1 - m5 and b1 - b5 separated by spaces.")
variables_list_strings = variables_input.split()
variables_list_floats = [float(item) for item in variables_list_strings]
variables_vector = np.array(variables_list_floats)

print(variables_vector)

np.set_printoptions(suppress=True)

# major loop for the code
iteration = 0
while iteration <= 1000000:
    # here I am getting each part of the list
    m1 = variables_vector[0]
    m2 = variables_vector[1]
    m3 = variables_vector[2]
    m4 = variables_vector[3]
    m5 = variables_vector[4]
    b1 = variables_vector[5]
    b2 = variables_vector[6]
    b3 = variables_vector[7]
    b4 = variables_vector[8]
    b5 = variables_vector[9]

    test_value = random.uniform(0, 5.5)

    if 0 <= test_value < 1.5:
        y = np.array([1, 0, 0, 0, 0])
    if 1.5 <= test_value < 2.5:
        y = np.array([0, 1, 0, 0, 0])
    if 2.5 <= test_value < 3.5:
        y = np.array([0, 0, 1, 0, 0])
    if 3.5 <= test_value < 4.5:
        y = np.array([0, 0, 0, 1, 0])
    if 4.5 <= test_value <= 5.5:
        y = np.array([0, 0, 0, 0, 1])

    h_hat = np.array([m1*test_value+b1, m2*test_value+b2, m3*test_value+b3, m4*test_value+b4, m5*test_value+b5])
    y_hat = softmax(h_hat)


    gradient_vector = np.array([test_value*(y_hat[0]-y[0]), test_value*(y_hat[1]-y[1]), test_value*(y_hat[2]-y[2]), test_value*(y_hat[3]-y[3]), test_value*(y_hat[4]-y[4]), y_hat[0]-y[0], y_hat[1]-y[1], y_hat[2]-y[2], y_hat[3]-y[3], y_hat[4]-y[4]])
    variables_vector -= 0.001 * gradient_vector


    if iteration == 1000000:
        water_tester = get_float_input("pick a number between 0 and 5.5 to test")
        test_value = water_tester
        if 0 <= test_value < 1.5:
            y = np.array([1, 0, 0, 0, 0])
        if 1.5 <= test_value < 2.5:
            y = np.array([0, 1, 0, 0, 0])
        if 2.5 <= test_value < 3.5:
            y = np.array([0, 0, 1, 0, 0])
        if 3.5 <= test_value < 4.5:
            y = np.array([0, 0, 0, 1, 0])
        if 4.5 <= test_value <= 5.5:
            y = np.array([0, 0, 0, 0, 1])

        h_hat = np.array([m1 * test_value + b1, m2 * test_value + b2, m3 * test_value + b3, m4 * test_value + b4, m5 * test_value + b5])
        y_hat = softmax(h_hat)
        print(y_hat)
        redo = get_float_input("1 for correct, 0 for incorrect")
        if redo == 0:
        iteration = 0

    iteration += 1

print(variables_vector)