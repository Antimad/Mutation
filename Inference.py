import numpy as np

# file = r"Data\#1--b_0.05_10-d_0.05_10-m_1e-3-p_1000.npz"

sample_properties = {
    "pop_size": 1000,
    "S_Length": 30,
    "Generations": 500,
    "Mutation Rate": 1e-3,
    "Selection": [0.05] * 10 + [0] * 10 + [-0.05] * 10
}
N = 1000


def sum_mutant_allele_sites(generation: np.array, size: np.array, empty_array: np.array):
    result = empty_array
    for i, seq in enumerate(generation):
        result += seq * size[i]
    return result


def calc_xij(generation, i, j, sz):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for idx, total in enumerate(column_sum):
        if total > 1:
            x_ij_count += sz[idx]

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
                x_ij = calc_xij(generation, i=i_idx, j=j_idx, sz=size)
                off_diagonal_covariance = (x_ij - (x_i_freq * x_j_freq))
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    temp_cov += covariance

    return temp_cov


def main(file_data):
    gen_data = np.load(file_data, allow_pickle=True)
    top = bottom = 0
    seq_l = sample_properties["S_Length"]
    regularization = np.identity(seq_l)/seq_l
    x_0 = sum_mutant_allele_sites(generation=gen_data["Sequence"][0],
                                  size=gen_data["Size"][0], empty_array=np.zeros(seq_l))
    for time, sequences in enumerate(gen_data["Sequence"]):
        size = gen_data["Size"][time]
        if time == 0:
            x_n = sum_mutant_allele_sites(generation=gen_data["Sequence"][time],
                                          size=size, empty_array=np.zeros(seq_l))
            delta_x_i = x_n/N - x_0 / N
        else:
            x_n_minus_1 = sum_mutant_allele_sites(generation=gen_data["Sequence"][time - 1],
                                                  size=gen_data["Size"][time - 1], empty_array=np.zeros(seq_l))
            x_n = sum_mutant_allele_sites(generation=gen_data["Sequence"][time],
                                          size=size, empty_array=np.zeros(seq_l))
            delta_x_i = x_n/N - x_n_minus_1/N
        covariance_matrix = covariance_builder(generation=sequences, size=size,
                                               dim=seq_l) + regularization
        c_ij_s_k = covariance_matrix.dot(sample_properties["Selection"])
        top_left = delta_x_i - c_ij_s_k
        inverted_covariance = np.linalg.inv(covariance_matrix)
        x_j = x_n/N
        top_right = inverted_covariance.dot(1 - (2 * x_j))

        top += top_left.dot(top_right)

        # bottom
        x_i = x_n/N
        bottom += (inverted_covariance.dot(1 - (2 * x_i))).dot((1 - (2 * x_j)))

    mu = top/bottom
    return mu
