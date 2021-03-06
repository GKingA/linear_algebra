import numpy as np
from numpy.polynomial import Polynomial
from norm import vector_norm


EPSILON = 1e-3
inf = 1e16


def print_pairs(values, vectors):
    for (value, vector) in zip(values, vectors):
        print(f"{value}: {[v for v in vector]}")


def check_symmetry(mtx):
    return (mtx.T == mtx).min()


def housholder_transformation(mtx):
    A = mtx
    E = np.identity(len(mtx))
    for i in range(0, len(mtx)-1):
        c = A[i][i]
        b = np.sign(-A[i+1, i]) * np.sqrt(np.sum(np.power(A[i+1:, i], 2)))
        Ae_Ae = np.array([0 for _ in range(i+1)] + [A[i+1][i] - b] + A[i+2:, i].tolist())
        norm_Ae_Ae = vector_norm(Ae_Ae, 2)
        u = (1 / norm_Ae_Ae) * Ae_Ae
        assert np.all(np.matmul(u[:, None].T, E[:, i]) == 0)
        T = np.identity(len(mtx)) - 2 * np.matmul(u[:, None], u[:, None].T)
        assert np.all(np.matmul(T, E[:, i]) == E[:, i])
        A = np.matmul(np.matmul(np.linalg.inv(T), A), T)
    return A


def create_strum_polynomial(mtx):
    polynomials = [1, Polynomial([-mtx[0][0], 1])]
    for i in range(1, len(mtx)):
        polynomials.append(Polynomial([-mtx[i][i], 1]) * polynomials[-1] - pow(mtx[i-1][i], 2) * polynomials[-2])
    return polynomials


def sign_changes(polynomials, value):
    prev_sign = (polynomials[0] >= 0) * 1
    diffs = 0
    for poly in polynomials[1:]:
        sign = (np.polyval(poly.coef[-1::-1], value) >= 0) * 1  # 0 if negative, 1 otherwise
        if prev_sign != sign:
            diffs += 1
        prev_sign = sign
    return diffs


def iterative_search_for_root(polynomials, small_end, large_end, no_elements):
    roots = []
    poly_small = np.polyval(polynomials[-1].coef[-1::-1], small_end)
    poly_large = np.polyval(polynomials[-1].coef[-1::-1], large_end)
    if abs(poly_small) <= EPSILON < abs(poly_large):
        roots.append(small_end)
    elif abs(poly_large) <= EPSILON < abs(poly_small):
        roots.append(large_end)
    elif abs(poly_large) <= EPSILON >= abs(poly_small):
        roots.append((large_end + small_end) / 2)
    while len(roots) < no_elements:
        no_down = abs(sign_changes(polynomials, small_end) - sign_changes(polynomials, (large_end + small_end) / 2))
        no_up = abs(sign_changes(polynomials, large_end) - sign_changes(polynomials, (large_end + small_end) / 2))
        if no_down > 0:
            roots += iterative_search_for_root(polynomials, small_end, (large_end + small_end) / 2, no_down)
        if no_up > 0:
            roots += iterative_search_for_root(polynomials, (large_end + small_end) / 2, large_end, no_up)
    return roots


def eigen_values(mtx):
    assert check_symmetry(mtx)
    mtx = housholder_transformation(mtx)
    polynomials = create_strum_polynomial(mtx)
    V_neg_inf = sign_changes(polynomials, -inf)
    V_pos_inf = sign_changes(polynomials, inf)
    V_zero = sign_changes(polynomials, 0)
    roots = []
    print(f"Number of roots: {abs(V_neg_inf - V_pos_inf)}\n\tNegative: {abs(V_neg_inf - V_zero)}\n\tPositive: {abs(V_pos_inf - V_zero)}")
    if abs(V_neg_inf - V_zero) > 0:
        roots += iterative_search_for_root(polynomials, -inf, 0, abs(V_neg_inf - V_zero))
    if abs(V_pos_inf - V_zero) > 0:
        roots += iterative_search_for_root(polynomials, inf, 0, abs(V_neg_inf - V_zero))
    return list(set(sorted(roots)))


def qr_decomposition(matrix):
    R = matrix
    Q = np.identity(len(matrix))
    for j in range(0, len(matrix)):
        for i in range(len(matrix) - 1, j, -1):
            Qn = np.identity(len(matrix))
            degree = np.arctan(R[i][j] / R[j][j])
            Qn[j][j] = np.cos(degree)
            Qn[i][i] = np.cos(degree)
            Qn[j][i] = np.sin(degree)
            Qn[i][j] = -np.sin(degree)
            Q = np.matmul(Qn, Q)
            R = np.matmul(Qn, R)
    return Q.T, R


def hessenberg_transformation(matrix):
    # Initialize
    Z = np.zeros(matrix.shape)
    F = np.zeros(matrix.shape)
    Z[0][0] = 1
    F[0][0] = matrix[0][0]
    for i in range(len(F)-1):
        F[i+1][i] = 1
    for i in range(1, len(Z)):
        Z[i][1] = matrix[i][0]
    # Go through columns
    for i in range(1, len(matrix)):
        AZ = np.matmul(matrix, Z)
        # Rows of column
        for j in range(i+1):
            F[j][i] = (1 / Z[j][j]) * (AZ[j][i] - sum([F[l][i] * Z[j][l] for l in range(j)]))
        ZF = np.matmul(Z, F)
        if i + 1 != len(Z):
            Z[:, i+1] = (AZ - ZF).T[i]
    #assert np.all(np.abs(np.matmul(matrix, Z) - np.matmul(Z, F)) < EPSILON)
    return F, Z


def eigenvector_for_hessenberg(F, eigen_value):
    if eigen_value.dtype == np.complex:
        y = np.zeros(len(F), dtype=np.complex)
    else:
        y = np.zeros(len(F))
    y[-1] = 1
    for i in range(len(y)-2, -1, -1):
        y[i] = eigen_value * y[i+1] - sum([F[i + 1][k] * y[k] for k in range(i, len(y))])
    #assert np.all(np.abs(eigen_value * y - np.dot(F, y)) <= EPSILON)
    return y


def eigen_vectors_from_hessenberg(F, Z, e_values, matrix):
    vectors = []
    for value in e_values:
        y = eigenvector_for_hessenberg(F, value)
        e_vector = np.matmul(Z, y)
        #assert np.all(np.abs(value * e_vector - np.dot(matrix, e_vector)) < EPSILON)
        vectors.append(e_vector)
    return vectors


def qr_algorithm(matrix):
    matrix0 = matrix
    for i in range(2*len(matrix)**2):
        Q, R = qr_decomposition(matrix0)
        matrix0 = np.matmul(R, Q)
    return matrix0


def get_eigenvalues(matrix):
    matrix0 = qr_algorithm(matrix)
    under_line = [i for i in range(len(matrix) - 1) if matrix0[i+1][i] > EPSILON]
    under_line += [u + 1 for u in under_line]
    if len(under_line) == 0:
        e_values = [matrix0[i][i] for i in range(len(matrix))]
    else:
        e_values = [matrix0[i][i] for i in range(len(matrix)) if i not in under_line]
        if len(under_line) == 2:
            from_ = min(under_line)
            to_ = max(under_line) + 1
            sub_matrix = matrix0[from_:to_, from_:to_]
            b = -(sub_matrix[0][0] + sub_matrix[1][1])+0J
            c = sub_matrix[0][0] * sub_matrix[1][1] - sub_matrix[0][1] * sub_matrix[1][0]
            e_values.append((-b - np.sqrt(b ** 2 - 4 * c)) / 2)
            e_values.append((-b + np.sqrt(b ** 2 - 4 * c)) / 2)
    return e_values


def rank_one_decomposition(matrix, u_shape, v_shape):
    if matrix.dtype == np.complex:
        U = np.zeros(u_shape, dtype=np.complex)
        Vt = np.zeros(v_shape, dtype=np.complex)
    else:
        U = np.zeros(u_shape)
        Vt = np.zeros(v_shape)
    # First column of U and first row of V
    for i in range(len(matrix)):
        U[i][0] = matrix[i][0]
        Vt[0][i] = matrix[0][i] / U[0][0]
    # Rest of the columns and rows
    for i in range(1, len(matrix[0])):
        for j in range(i, len(matrix)):
            U[j][i] = matrix[j][i] - sum([U[j][k] * Vt[k][i] for k in range(i)])
            Vt[i][j] = (1 / U[i][i]) * (matrix[i][j] - sum([U[i][k] * Vt[k][j] for k in range(j)]))
    #assert np.all(np.abs(np.matmul(U, Vt) - matrix) < EPSILON)
    return U, Vt


def get_eigenvectors(matrix, e_values):
    xs = []
    for e_value in e_values:
        # Ab = np.concatenate([matrix - e_value * np.identity(len(matrix)), np.zeros(len(matrix))[:, None]], axis=1)
        Ab = matrix - e_value * np.identity(len(matrix))
        if Ab.dtype == np.complex:
            x = np.zeros(len(Ab), dtype=np.complex)
        else:
            x = np.zeros(len(Ab))
        _, Vt = rank_one_decomposition(Ab, (len(matrix), len(matrix[0])), (len(Ab), len(Ab[0])))
        # x[-1] = Vt[-1][-2]
        x[-1] = Vt[-1][-1]
        for i in range(len(x) - 2, -1, -1):
            x[i] = -sum([Vt[i][j] * x[j] for j in range(i, len(x))])
        xs.append(x)
        #assert np.all(np.abs(e_value * x - np.dot(matrix, x)) < EPSILON)
    return xs