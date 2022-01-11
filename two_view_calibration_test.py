import numpy as np

from test import test, run_tests

import li_utils

import two_view_calibration

@test
def estimated_fundemental_matrix():
    P_1 = np.random.rand(3, 4)
    P_2 = np.random.rand(3, 4)

    num_pts = 100

    X = np.random.rand(4, num_pts)
    X = X/X[-1]

    x1 = P_1 @ X
    x2 = P_2 @ X

    K1, R1, p1 = li_utils.get_projection_product_matricies(P_1)
    K2, R2, p2 = li_utils.get_projection_product_matricies(P_2)

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    F = two_view_calibration.estimate_fundamental_matrix(x1, x2)

    # property of essential matrices x''.T @ E x' = 0
    for i in range(num_pts):
        assert (np.abs(x2[:,i] @ F @ x1[:,i][...,None]) < 1e-8).all() , "fundamental matrix does not have defining property" 

    E = K2.T @ F @ K1
    assert np.allclose(2*E @ E.T @ E - np.trace(E@E.T)*E, 0), "essential matrix not dirived from fundamental matrix"

@test
def estimated_essential_matrix():
    P_1 = np.random.rand(3, 4)
    P_2 = np.random.rand(3, 4)

    num_pts = 100

    X = np.random.rand(4, num_pts)
    X = X/X[-1]

    x1 = P_1 @ X
    x2 = P_2 @ X

    K1, R1, p1 = li_utils.get_projection_product_matricies(P_1)
    K2, R2, p2 = li_utils.get_projection_product_matricies(P_2)

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    kx1 = K1_inv @ x1
    kx2 = K2_inv @ x2

    E = two_view_calibration.estimate_essential_matrix(kx1, kx2)

    # property of essential matrices x''.T @ E x' = 0
    for i in range(num_pts):
        assert np.allclose(kx2[:,i] @ E @ kx1[:,i][...,None], 0), "essential matrix does not have defining property" 
    assert np.allclose(2*E @ E.T @ E - np.trace(E@E.T)*E, 0), "essential matrix does not have correct property"

@test
def get_canidate_essential_product_matrcies():
    P_1 = np.random.rand(3, 4)
    P_2 = np.random.rand(3, 4)

    X = np.random.rand(4, 10)
    X = X/X[-1]

    x1 = P_1 @ X
    x2 = P_2 @ X

    K1, R1, p1 = li_utils.get_projection_product_matricies(P_1)
    K2, R2, p2 = li_utils.get_projection_product_matricies(P_2)

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    kx1 = K1_inv @ x1
    kx2 = K2_inv @ x2

    E = two_view_calibration.estimate_essential_matrix(kx1, kx2)
    S_a, S_b, R_a, R_b, _ = two_view_calibration.get_canidate_essential_product_matrcies(E)

    Es = [S_a@R_a, -S_a@R_b, S_b@R_b, -S_b@R_a]

    # we cannot compare to the original E because the U and V of the svd of E are normalized
    for E_hat in Es:
        # since we normlaize the estimated matrix E_hat, 
        # E_hat could be a scalar multiple of E (should be either 1 or -1 in this case)
        # since every element should have the same scale
        # we only have to check the first element
        scale = E[0,0]/E_hat[0,0]
        assert np.allclose(E, scale*E_hat), "essential matrix product matricies aren't consistent with eachother"

@test
def get_essential_product_matrcies():
    np.random.seed(19)

    p1 = np.random.rand(3)[...,None]
    p2 = np.random.rand(3)[...,None]

    # negitive camera constant... so its pointing forward
    # Why is this important?
    K = li_utils.I_3 @ li_utils.rotate3d_around_z_180

    R1 =li_utils.I_3
    w = np.random.rand(3)
    R2 =li_utils.rotation_angles_to_matrix(w)

    # w1 = np.random.rand(3)
    # w2 = np.random.rand(3)
    # R1 =li_utils.rotation_angles_to_matrix(w1)
    # R2 =li_utils.rotation_angles_to_matrix(w2)

    P_1 = li_utils.product_matricies_to_projection_matrix(K, R1, p1)
    P_2 = li_utils.product_matricies_to_projection_matrix(K, R2, p2)

    X = np.random.rand(4, 10)
    X = X/X[-1]

    X += np.array([-1,-1,2,0])[...,None]

    K1, R1, p1 = li_utils.get_projection_product_matricies(P_1)
    K2, R2, p2 = li_utils.get_projection_product_matricies(P_2)

    x1 = P_1 @ X
    x2 = P_2 @ X

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    kx1 = K1_inv @ x1
    kx1 = li_utils.to_euclid_coords(kx1)
    kx2 = K2_inv @ x2
    kx2 = li_utils.to_euclid_coords(kx2)

    E = two_view_calibration.estimate_essential_matrix(kx1, kx2)

    # Start of actual test:

    P_2_est, _ = two_view_calibration.get_projection_matrix_and_product_matricies_from_essential_matrix(E, kx1.T[5], kx2.T[5])

    P_1_est = two_view_calibration.P_base

    X_est = li_utils.triangulate_points(P_1_est, P_2_est, kx1, kx2)

    scale, R_sc, T = li_utils.iterative_closest_point_with_scale(X[:3],X_est)

    X_sc = scale*R_sc@X_est + T

    R1_est, p1_est = li_utils.get_rotation_and_position_from_calibrated_projection_matrix(P_1_est)
    R2_est, p2_est = li_utils.get_rotation_and_position_from_calibrated_projection_matrix(P_2_est)

    assert np.allclose(R2, R2_est@R_sc.T), "relative camera rotation incorrect"
    assert np.allclose(p2, scale*R_sc@p2_est + T), "relative camera location incorrect"
    assert np.allclose(X[:3], X_sc), "estimated points do not match up"


@test
def get_calibrated_camera_location_relative_to_first():
    # np.random.seed(40)

    p1 = np.array([0,0,0])[...,None]
    p2 = np.array([3,1,0])[...,None]

    # negitive camera constant... so its pointing forward
    # Why is this important?
    #li_utils.I_3 @ li_utils.rotate3d_around_z_180
    K = np.diag(np.random.rand(3))
    K /= K[-1,-1]

    w1 = np.random.rand(3)*0.1
    w2 = np.random.rand(3)*0.1
    R1 =li_utils.rotation_angles_to_matrix(w1)
    R2 =li_utils.rotation_angles_to_matrix(w2)

    P_1 = li_utils.product_matricies_to_projection_matrix(K, R1, p1)
    P_2 = li_utils.product_matricies_to_projection_matrix(K, R2, p2)

    num_points = 15
    X = np.random.rand(4, num_points)
    X = X/X[-1]

    X += np.array([-1,-1,2,0])[...,None]

    K1, R1, p1 = li_utils.get_projection_product_matricies(P_1)
    K2, R2, p2 = li_utils.get_projection_product_matricies(P_2)

    x1 = P_1 @ X
    x2 = P_2 @ X
  
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    kx1 = K1_inv @ x1
    kx1 = li_utils.to_euclid_coords(kx1)
    kx2 = K2_inv @ x2
    kx2 = li_utils.to_euclid_coords(kx2)

    # Start of actual test

    P_1_est, P_2_est = two_view_calibration.get_calibrated_camera_location_relative_to_first(kx1, kx2,iters=10000)

    X_est = li_utils.triangulate_points(P_1_est, P_2_est, kx1, kx2)

    scale, R_sc, T = li_utils.iterative_closest_point_with_scale(X[:3],X_est)

    X_sc = scale*R_sc@X_est + T
    R1_est, p1_est = li_utils.get_rotation_and_position_from_calibrated_projection_matrix(P_1_est)
    R2_est, p2_est = li_utils.get_rotation_and_position_from_calibrated_projection_matrix(P_2_est)

    assert np.allclose(R1, R_sc.T@R1_est), "first relative camera rotation incorrect"
    assert np.allclose(R2, R2_est@R_sc.T), "second relative camera rotation incorrect"
    assert np.allclose(p1, scale*R_sc@p1_est + T), "first relative camera location incorrect"
    assert np.allclose(p2, scale*R_sc@p2_est + T), "second relative camera location incorrect"
    
    assert np.allclose(X[:3], X_sc), "estimated matricies incorrect"

    

if __name__ == '__main__':
    run_tests()

    