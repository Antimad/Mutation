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

sequence_size_matrix = sample[0].to_numpy()

"""
Adding the n_th index in each array group together by using
    np.einsum('ij->j, Array)
"""

index = 1
X_i = np.einsum('ij->j', sequence_size_matrix[0][index])


# Looping through all indices will

"""
def calc_xij(generation, i, j):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for total in column_sum:
        if total > 1:
            x_ij_count += 1

    return x_ij_count


def covariance_builder(generation: np.array, size: np.array):
    covariance_matrix = np.zeros((50, 50))
    generation_with_size = (generation.T * size).T
    generation_with_size = sum(generation_with_size)
    covariance = []

    for i_idx, x_i_sum in enumerate(generation_with_size):
        covariance_list = []
        for j_idx, x_j_sum in enumerate(generation_with_size):
            if i_idx == j_idx:
                covariance_diagonal = (x_i_sum * (1 - x_i_sum)) / sample_properties["pop_size"]
                covariance_list.append(covariance_diagonal)
            else:
                x_ij = calc_xij(generation, i=i_idx, j=j_idx)
                off_diagonal_covariance = (x_ij - (x_i_sum * x_j_sum)) / sample_properties["pop_size"]
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    covariance_matrix += covariance

    return covariance


seq = np.array(sequence_size_matrix[0][2])
Size = np.array(sequence_size_matrix[1][2])
Covariance = covariance_builder(generation=seq, size=Size)

Xi = np.empty(shape=[0, 50])
for idx, sequence in enumerate(sequence_size_matrix[0]):
    # delta_x_i - Summation(C_ik * S_k)
    # TODO: delta_X_i =
    Covariance = covariance_builder(generation=np.array(sequence), size=np.array(sequence_size_matrix[1][idx]))
    Sum_C_ik = np.eimsum('ik->k', Covariance)
    Sum_C_ik_S = Sum_C_ik * sample_properties["Beneficial_Deleterious Fitness"]
    # end

    # C_ij^(-1)(1-2X_j)

    # Xi = np.append(Xi, np.matrix(X_i_einsum), axis=0)
"""