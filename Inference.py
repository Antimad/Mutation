import numpy as np
import pandas as pd

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

sequence_size_matrix = sample[0].to_numpy()


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

for idx, sequence in enumerate(sequence_size_matrix[0]):
    """
    The main loop handles time, as each sequence is a point(generation) in time. 
    einsum is used to handle the summations of indices i & j.
    """
    """
    The equation has indices i & j, for sites i & j.
    Since I'll be using einsum for the summations of i & j, I will reference 
    """
    # delta_x_i - Summation(C_ik * S_k)
    delta_X_i = sequence[-1][idx] - sequence[0][idx]
    Covariance = covariance_builder(generation=np.array(sequence), size=np.array(sequence_size_matrix[1][idx]))
    Sum_C_ik = np.eimsum('ik->k', Covariance)
    Sum_C_ik_S = Sum_C_ik * sample_properties["Beneficial_Deleterious Fitness"]
    # end

    # C_ij^(-1)(1-2X_j)
    inverse_Covariance = np.linalg.inv(Covariance)  # TODO: Fix, inverse D.N.E.

    # It's odd that they are the same, doubt...
    x_j = np.einsum('ij->j', sequence)[idx]
    x_i = np.einsum('ij->j', sequence)[idx]

    one_minus_2xj = (1 - (2 * x_j))
    one_minus_2xi = (1 - (2 * x_i))


