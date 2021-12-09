import numpy as np
from eigen import get_eigenvectors, get_eigenvalues


if __name__ == '__main__':
    m = np.array([[3, -1, 2, 7], [1, 2, 0, -1], [4, 2, 1, 1], [2, -1, -2, 2]])
    values = get_eigenvalues(m)
    vectors = get_eigenvectors(m, values)
    for (value, vector) in zip(values, vectors):
        print(f"{value}: {vector}")
