import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

file = r"C:\Users\unwae\PycharmProjects\UCR_W21-Wright_Fisher_Model\Sample.json"

sample = pd.read_json(file)

sample_properties = {
    "pop_size": 1000,
    "S_Length": 50,
    "Generations": 1000,
    "Mutation Rate": 0.001,
    "Beneficial_Deleterious Mutation": 10,
    "Beneficial_Deleterious Fitness": 0.03
}
N = 1000


def sum_mutant_allele_sites(generation: np.array, size: np.array, empty_array: np.array):
    result = empty_array
    for i, seq in enumerate(generation):
        result += seq * size[i]
    return result


def calc_xij(generation, i, j):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for total in column_sum:
        if total > 1:
            x_ij_count += 1

    return x_ij_count / N


def covariance_builder(generation: np.array, size: np.array, dim: int):
    temp_cov = np.zeros((dim, dim))
    generation_with_size = sum_mutant_allele_sites(generation=generation, size=size, empty_array=np.zeros(dim))
    covariance = []
    for i_idx, x_i_sum in enumerate(generation_with_size):
        covariance_list = []
        for j_idx, x_j_sum in enumerate(generation_with_size):
            x_i_freq = x_i_sum / N
            x_j_freq = x_j_sum / N
            if i_idx == j_idx:
                covariance_diagonal = (x_i_freq * (1 - x_i_freq))
                covariance_list.append(covariance_diagonal)
            else:
                x_ij = calc_xij(generation, i=i_idx, j=j_idx)
                off_diagonal_covariance = (x_ij - (x_i_freq * x_j_freq))
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    temp_cov += covariance

    return temp_cov


def main(gen_data):
    mu = 0
    regularization = 1 / (sample_properties["pop_size"])
    for time, sequences in enumerate(gen_data["Sequence"]):
        """
        The main loop handles time, as each sequence is a point(generation) in time. 
        einsum is used to handle the summations of indices i & j.
        """
        """
        The equation has indices i & j, for sites i & j.
        Since I'll be using einsum for the summations of i & j, I will reference 
        """
        covariance = covariance_builder(generation=np.array(sequences), size=np.array(gen_data["Size"][time]))
        sum_c_ik = np.einsum('ik->k', covariance)
        sum_c_ik_s = sum_c_ik * sample_properties["Beneficial_Deleterious Fitness"]
        covariance[np.diag_indices_from(covariance)] += regularization  # Regularization
        inverse_covariance = np.linalg.inv(covariance)
        top = bottom = 0
        for idx, sequence in enumerate(sequences):
            """
            This loop handle the iteration of each sequence in the generation (time) loop above.
            """
            delta_x_i = sequence[0] - sequence[-1]  # Should this be the other way around?
            delta_xi_minus_sum_cs = delta_x_i - sum_c_ik_s

            for index_i, site_i in enumerate(sequence):
                one_minus_2xi = (1 - (2 * site_i))

                for index_j, site_j in enumerate(sequence):
                    one_minus_2xj = (1 - (2 * site_j))
                    inv_c_times_1_2_x_j = (inverse_covariance[index_i][index_j] * one_minus_2xj)
                    top += (delta_xi_minus_sum_cs[index_i] * inv_c_times_1_2_x_j)
                    bottom += (one_minus_2xi * inv_c_times_1_2_x_j)
        mu = top/bottom
    return mu


results = []
index = 0
"""
for root, dirs, files in os.walk("Data", topdown=False):
    for file in files:
        index += 1
        data = np.load(os.path.join(root, file), allow_pickle=True)
        result = main(gen_data=data)
        results.append(result)
        print("Result: {0}, index: {0}".format(result, index))


plt.hist(x=results)
plt.ylabel("Mutation Rates")
plt.title("Mutation Rate Distribution")

plt.savefig("Mutation Distribution")
"""