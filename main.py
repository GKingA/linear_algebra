import numpy as np
from eigen import get_eigenvectors, get_eigenvalues, hessenberg_transformation, eigen_vectors_from_hessenberg, print_pairs, eigen_values
import time


if __name__ == '__main__':
    m = np.array([[3, -1, 2, 7], [1, 2, 0, -1], [4, 2, 1, 1], [2, -1, -2, 2]])
    print(f"Matrix:\n{m}")
    #m = np.random.rand(10, 10)
    #m = np.array([[1, 1, -1, 2], [1, 2, -2, 1], [2, 3, -1, 2], [1, 2, -1, 3]])
    n = time.time()
    m_values = get_eigenvalues(m)
    print(time.time()-n)
    n = time.time()
    F, Z = hessenberg_transformation(m)
    F_values = get_eigenvalues(F)
    print(time.time()-n)
    n = time.time()
    m_vectors = get_eigenvectors(m, m_values)
    print(time.time()-n)
    n = time.time()
    F_vectors = eigen_vectors_from_hessenberg(F, Z, F_values, m)
    print(time.time()-n)
    print("Eigen value - eigen vector pairs calculated with QR and ROD:")
    print_pairs(m_values, m_vectors)
    print("Eigen value - eigen vector pairs calculated with Hessenberg:")
    print_pairs(F_values, F_vectors)
    print("Eigen value - eigen vector pairs with numpy:")
    val, vec = np.linalg.eig(m)
    print_pairs(val, vec)

    symmetric_matrix = np.array([[1, 2, 3, 4, 7], [2, 1, 4, 5, 8], [3, 4, 1, 6, 9], [4, 5, 6, 1, 10], [7, 8, 9, 10, 1]])
    print(f"Eigen value - eigen vector pairs of the symmetrical matrix\n{symmetric_matrix}:")
    sym_values = eigen_values(symmetric_matrix)
    sym_vectors = get_eigenvectors(symmetric_matrix, sym_values)
    print_pairs(sym_values, sym_vectors)
