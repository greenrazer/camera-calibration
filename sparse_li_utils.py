import scipy.sparse
import scipy.sparse.linalg

def sparse_solve_linear_system(sparse_A, b):
    return scipy.sparse.linalg.spsolve(sparse_A, b)

def sparse_inv(sparse_A):
    return scipy.sparse.linalg.inv(sparse_A)

def make_sparse_zeros(m, n):
    return scipy.sparse.lil_matrix((m,n))

def convert_to_csr_sparse_matrix_for_computation(M):
    return M.tocsc()

