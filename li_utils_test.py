import numpy as np

import li_utils

DIST = .33
real_points_m = np.array([
    [0,0,0], # White
    [DIST,0,0], # Yellow
    [-DIST,0,0], # Blue
    [0,DIST,0], # Purple
    [0,-DIST,0], # Silver
    [0,0,DIST], # Red
    [0,0,-DIST], # Orange
])

points1 = np.array([(0.515, 0.4975), (0.40375, 0.835), (0.59125, 0.25), (0.86375, 0.57875), (0.2075, 0.41625), (0.52, 0.64), (0.49625, 0.2525)])
points2 = np.array([[0.47166667,0.4475],[0.925,0.18], [0.03333333 ,0.69625], [0.83833333 ,0.79], [0.20833333 ,0.21625], [0.21333333 ,0.12], [0.62,0.56]])
points3 = np.array([(0.495, 0.59), (0.58, 0.87125), (0.43625, 0.37875), (0.47625, 0.69375), (0.5275, 0.42), (0.2125, 0.6575), (0.7525, 0.545)])
points4 = np.array([(0.51625, 0.59125), (0.525, 0.655), (0.50875, 0.4925), (0.51, 0.515), (0.53125, 0.67625), (0.39625, 0.605), (0.64, 0.58125)])
points5 = np.array([(0.53375, 0.5675), (0.5925, 0.665), (0.49, 0.48625), (0.5425, 0.66125), (0.52125, 0.45375), (0.4, 0.615), (0.6625, 0.52625)])

tests = []

def test(func):
    def wrapper():
        print(f"Starting test \"{func.__name__}\"...")
        passed = True
        try:
            temp_passed = func()
            if temp_passed is not None:
                passed = temp_passed
        except AssertionError as e:
            print(" L x Test Failed with error : ", e)
        else:
            if passed:
                print(" L o Test Passed!")
            else:
                print(" L x Test Failed.")
    tests.append(wrapper)
    return wrapper


def test_calibration(func):
    def wrapper():
        return func(real_points_m, points1)
    wrapper.__name__ = func.__name__
    return test(wrapper)

def test_convergence(func):
    trys = 5
    def wrapper():
        A = np.array([[1.4395939  ,0.08200007 ,0.46280866 ,1.88044163],
                    [1.87785976 ,0.39312279 ,1.41691326 ,1.17382653],
                    [0.48335329 ,1.98663142 ,0.4507917  ,1.        ]])
        X_inp = np.array([[0.23419559 ,0.98057389 ,0.45490408],
                [0.00511944 ,0.40741681 ,0.56169471],
                [0.78121543 ,0.02989912 ,0.51052407],
                [0.0893092  ,0.30371104 ,0.12190221],
                [0.30338405 ,0.71321715 ,0.27758328],
                [0.69770992 ,0.22958662 ,0.86197783],
                [0.33185928 ,0.11562294 ,0.88173284],
                [0.66958615 ,0.73145093 ,0.6069468 ],
                [0.91435967 ,0.92104129 ,0.29118183],
                [0.80054708 ,0.65589752 ,0.52935   ],
                [0.01099736 ,0.53297328 ,0.7255345 ],
                [0.42717228 ,0.87169759 ,0.67448424]])
        X_out = np.array([[0.768002   ,0.80937282],
                [1.0562248  ,1.03603211],
                [1.94572632 ,2.0250001 ],
                [1.22853605 ,0.96013768],
                [0.93136755 ,0.8990428 ],
                [1.51363125 ,1.73958576],
                [1.55278858 ,1.72960035],
                [1.0442163  ,1.17321787],
                [1.00118851 ,1.07714884],
                [1.13763417 ,1.25829155],
                [0.95172254 ,1.01707066],
                [0.88796851 ,1.0098996 ]])
        if func(A, X_inp, X_out) is False:
            return False
        for i in range(trys):
            A = np.random.rand(3,4)
            A /= np.linalg.norm(A)
            A /= A[-1, -1]

            X = np.random.rand(3,12)
            X = li_utils.to_homo_coords(X)

            X_inp = li_utils.to_euclid_coords(X,   entire=False).T
            X_out = li_utils.to_euclid_coords(A@X, entire=False).T
            
            try:
                return func(A, X_inp, X_out)
            except AssertionError:
                print(f" Failed try {i+1} of {trys}, attempting again...")

        return func(A, X_inp, X_out)

    wrapper.__name__ = func.__name__
    return test(wrapper)

@test_calibration
def camera_matrix_decomposition(real_points, screen_points):
    P_before = li_utils.dlt(real_points, screen_points)
    K, R, pos = li_utils.get_projection_product_matricies(P_before)
    P_after = li_utils.product_matricies_to_projection_matrix(K, R, pos)

    P_after /= P_after[-1, -1]

    assert np.allclose(P_before, P_after), "Projection matrix decomp not working"

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

@test_calibration
def normalization(real_points, screen_points):
    normalization_helper(real_points)
    normalization_helper(screen_points)

@test_calibration
def camera_projection_jacobian(real_points, screen_points):
    P = li_utils.dlt(real_points, screen_points)

    X = li_utils.to_homo_coords(real_points.T)
    x = li_utils.to_homo_coords(screen_points.T)

    J1, _ = li_utils.camera_projection_levenberg_marquardt_jacobian_and_residual(P, X, x)

    def resi(P):
        return li_utils.camera_project_points_operation(P, X) - x

    J2 = li_utils.numerical_jacobian(resi, P)

    epsilon = 1e-7
    assert (abs(J1 - J2) < epsilon).all(), f"Camera projection jacobian not roughy equal to the numerical estimation within {epsilon}"

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

@test_convergence
def direct_linear_transform(A, real_points, screen_points):
    # If P is the best estimation of the projection matrix using the dlt
    # If we perform the dlt again using the estimated screen points the error
    # should be close to zero and the new estimated projection should be roughly equal

    A_new = li_utils.dlt(real_points, screen_points)

    assert np.allclose(A, A_new), "DLT not working"

@test_convergence
def camera_projection_levenberg_marquardt(A, real_points, screen_points):
    A_new = li_utils.camera_projection_levenberg_marquardt(real_points, screen_points, A)

    assert np.allclose(A, A_new), "levenberg marquardt algorithm diverging from optimum"

    A_adj = A + np.random.rand(*A.shape)*1e-7

    A_new = li_utils.camera_projection_levenberg_marquardt(real_points, screen_points, A_adj, iters=24)
    assert (np.abs(A - A_new) < 1e-5).all(), "levenberg marquardt does not converge to optimum from slightly different start"

@test_convergence
def numerical_camera_projection_levenberg_marquardt(A, real_points, screen_points):
    A_new = li_utils.numerical_camera_projection_levenberg_marquardt(real_points, screen_points, A)

    assert np.allclose(A, A_new), "levenberg marquardt algorithm diverging from optimum"

    assemble_func = li_utils.assemble_feature_vector
    def test_assemble_feature_vector(P, include_K=True):
        a = assemble_func(P)
        return a + np.random.rand(*a.shape)*1e-6
    li_utils.assemble_feature_vector = test_assemble_feature_vector

    A_new = li_utils.numerical_camera_projection_levenberg_marquardt(real_points, screen_points, A, iters=15)

    assert (np.abs(A - A_new) < 1e-4).all(), "levenberg marquardt does not converge to optimum from slightly different start"

@test_convergence
def calibrate_camera(A, real_points, screen_points):
    A_new, _ = li_utils.calibrate_camera(real_points, screen_points)
    assert (np.abs(A - A_new) < 1e-4).all(), "Calibrate Camera Not Converging"

@test_convergence
def calibrate_camera_const_internals(A, real_points, screen_points):
    real, avg_real, scale_real = li_utils.normalize_points(real_points)
    screen, avg_screen, scale_screen = li_utils.normalize_points(screen_points)
    real_norm_matrix  = li_utils.construct_normalization_matrix(4, avg_real, scale_real)
    real_norm_matrix_inv = li_utils.np.linalg.inv(real_norm_matrix)
    screen_norm_matrix = li_utils.construct_normalization_matrix(3, avg_screen, scale_screen)

    A_norm = screen_norm_matrix @ A @ real_norm_matrix_inv
    A_norm /= A_norm[-1,-1]

    K, _, _ = li_utils.get_projection_product_matricies(A_norm)

    A_new, _ = li_utils.calibrate_camera_const_internals(real_points, screen_points, K, A_norm)

    assert (np.abs(A - A_new) < 1e-4).all(), "Calibrate Camera Not Converging"


if __name__ == '__main__':
    for test in tests:
        test()