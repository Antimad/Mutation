import numpy as np
import pandas as pd
import numpy as np

file = r"C:\Users\unwae\PycharmProjects\UCR_W21-Wright_Fisher_Model\Sample.json"

sample = pd.read_json(file)

sample_properties = {
    "pop_size": 1000,
    "S_Length": 50,
    "Generations": 100,
    "Mutation Rate": 0.001,
    "Beneficial_Deleterious Mutation": 10,
    "Beneficial_Deleterious Fitness": 0.03

}

A = np.array([0, 1, 2])
A_2D = np.array([[1, 1, 1],
                 [2, 2, 2],
                 [5, 5, 5]])

B = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

"""
The traditional way multiplies the index throughout the entire row
taking more steps, and requires more resources to execute
"""

traditional = (A[:, np.newaxis] * B).sum(axis=1)
answer = np.einsum('i, ij->i', A, B)  # einsum

"""
Steps:

1. Select the first index X_1 in each sequence from every generation (use first pop group)
2. Subtract sum of the (Covariance*Selection) @ index n. 
        Where n=i=1 (Visual needed) 
        
"""


# TODO: transform Sample df to numpy matrix array.

# TODO: Use numpy matrix array with einsum to get delta_X_1
