import numpy as np
from eigen import get_eigenvectors, get_eigenvalues, hessenberg_transformation, eigen_vectors_from_hessenberg, print_pairs
import time


if __name__ == '__main__':
    m = np.array([[3, -1, 2, 7], [1, 2, 0, -1], [4, 2, 1, 1], [2, -1, -2, 2]])
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
