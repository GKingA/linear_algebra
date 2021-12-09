import numpy as np
from eigen import get_eigenvalues


def vector_norm(vector, p=2):
    if p == 0:
        return 1
    if p == np.inf:
        return np.max(np.abs(vector))
    return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)


def frobenius_norm(matrix):
    return np.power(np.sum(np.power(matrix, 2)), 1/2)


def other_matrix_norm(matrix):
    return len(matrix) * np.max(np.abs(matrix))


def one_norm(matrix):
    return np.max(np.sum(np.abs(matrix), axis=0))


def infinity_norm(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))


def two_norm(matrix):
    return np.sqrt(np.max(get_eigenvalues(np.matmul(matrix.T, matrix))))
