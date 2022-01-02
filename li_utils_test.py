import numpy as np

import li_utils

from test import test, run_tests

@test
def camera_matrix_decomposition():
    A = np.random.rand(3,4)
    A = A/A[-1,-1]
    K, R, pos = li_utils.get_projection_product_matricies(A)
    A_after = li_utils.product_matricies_to_projection_matrix(K, R, pos)

    A_after /= A_after[-1, -1]

    assert np.allclose(A_after, A), "Projection matrix decomp not working"

def normalization_helper(pts):
    dims = pts.shape[1]
    new_points, avg, scale = li_utils.normalize_points(pts)

    avg_pt = np.average(new_points, axis=0)
    assert np.allclose(avg_pt, 0), "Average point is not zero."

    distances = np.linalg.norm(new_points, ord=dims, axis=1)
    avg_dist = np.average(distances) 
    assert np.allclose(avg_dist, np.sqrt(dims)), f"Average of distances is {np.round(avg_dist,5)} instead of sqrt({dims})."

    old_points = li_utils.unnormalize_points(new_points, avg, scale)
    assert np.allclose(old_points, pts), "Reverse normalization failed."

    norm_mat = li_utils.construct_normalization_matrix(dims+1, avg, scale)
    norm_mat_inv = np.linalg.inv(norm_mat)

    X = li_utils.to_homo_coords(pts.T)
    trans_X = norm_mat@X
    trans_points = li_utils.cut_last_row(trans_X).T

    assert np.allclose(trans_points, new_points), "normalization matrix incorrect"

    back_X = norm_mat_inv@trans_X
    back_points = li_utils.cut_last_row(back_X).T

    assert np.allclose(back_points, pts), "inverse normalization matrix incorrect"

@test
def normalization():
    for i in range(3):
        pts = np.random.rand(100, i+3)
        normalization_helper(pts)

@test
def axis_angle():
    w1 = (np.random.rand(3)-0.5)*2*np.pi

    while np.linalg.norm(w1) > np.pi:
        w1 = (np.random.rand(3)-0.5)*2*np.pi

    R1 = li_utils.rotation_angles_to_matrix(w1)
    w1_again = li_utils.rotation_matrix_to_angles(R1)

    assert np.allclose(w1, w1_again), "Applying inverse to matrix not working."

    w_rot = li_utils.rotation_matrix_to_angles(li_utils.rotate3d_around_z_180 @ R1)
    R1_rot = li_utils.rotation_angles_to_matrix(w_rot)

    assert np.allclose(li_utils.rotate3d_around_z_180 @ R1, R1_rot), "axis angle wrong after rotation"
    
    R1_again = li_utils.rotate3d_around_z_180 @ R1_rot

    assert np.allclose(R1, R1_again), "rotating back does after axis angle does not give correct value"

@test
def vec():
    A = np.random.rand(3,4)
    a = li_utils.vec(A)
    A_new = li_utils.unvec(a)

    assert np.allclose(A, A_new), "vec or unvec not working for matrix"

    shape = (3,4,5)
    A = np.random.rand(*shape)
    a = li_utils.vec(A)
    A_new = li_utils.unvec(a, shape)

    assert np.allclose(A, A_new), "vec or unvec not working for 3 rank tensor"

    shape = (3, 4, 5, 6)
    A = np.random.rand(*shape)
    a = li_utils.vec(A)
    A_new = li_utils.unvec(a, shape)

    assert np.allclose(A, A_new), "vec or unvec not working for 4 rank tensor"

@test
def rotation_around_z():
    A1 = np.random.rand(3,3)
    A2 = np.random.rand(3,3)

    A_prod = A2 @ A1

    R1 = li_utils.rotate3d_around_z_180 @ A1
    R2 = A2 @ li_utils.rotate3d_around_z_180

    A_R_prod = R2 @ R1

    assert np.allclose(A_prod, A_R_prod), "Rotation matricies don't cancel out"

@test
def numerical_jacobian():
    def scale_by_3(X):
        return 3*X

    n = 9
    vec_n = np.random.rand(n)
    J = li_utils.numerical_jacobian(scale_by_3, vec_n)

    assert np.allclose(3*np.identity(n), J), "scale jacobian failed."

    def sq_matrix(M):
        return M @ M

    n = 3
    nxn = np.random.rand(n,n)
    J = li_utils.numerical_jacobian(sq_matrix, nxn)

    identity_n = np.identity(n)

    J_true = np.kron(identity_n, nxn) + np.kron(nxn.T, identity_n)

    assert np.allclose(J, J_true), "square jacobian failed"

if __name__ == '__main__':
    run_tests()