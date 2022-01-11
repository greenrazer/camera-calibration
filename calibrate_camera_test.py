import numpy as np

import li_utils

import calibrate_camera

from test import test, run_tests

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
        assert func(A, X_inp, X_out) is not False, "Convergence not working on known working example"
        for i in range(trys):
            A = np.random.rand(3,4)
            A /= np.linalg.norm(A)
            A /= A[-1, -1]

            X = np.random.rand(3,12)
            X = li_utils.to_homo_coords(X)

            x = A@X
            x = x/x[-1]

            X_inp = li_utils.to_euclid_coords(X, entire=False).T
            X_out = li_utils.to_euclid_coords(x, entire=False).T
            
            try:
                return func(A, X_inp, X_out)
            except AssertionError as e:
                print(f" Failed: {e}... try {i+1} of {trys}, attempting again...")

        return func(A, X_inp, X_out)

    wrapper.__name__ = func.__name__
    return test(wrapper)

@test_calibration
def camera_projection_jacobian(real_points, screen_points):
    P = calibrate_camera.dlt(real_points, screen_points)

    X = li_utils.to_homo_coords(real_points.T)
    x = li_utils.to_homo_coords(screen_points.T)

    J1, _ = calibrate_camera.camera_projection_levenberg_marquardt_jacobian_and_residual(P, X, x)

    def resi(P):
        return calibrate_camera.camera_project_points_operation(P, X) - x

    J2 = li_utils.numerical_jacobian(resi, P)

    epsilon = 1e-7
    assert (abs(J1 - J2) < epsilon).all(), f"Camera projection jacobian not roughy equal to the numerical estimation within {epsilon}"

@test_convergence
def direct_linear_transform(A, real_points, screen_points):
    # If P is the best estimation of the projection matrix using the dlt
    # If we perform the dlt again using the estimated screen points the error
    # should be close to zero and the new estimated projection should be roughly equal

    A_new = calibrate_camera.dlt(real_points, screen_points)

    assert np.allclose(A, A_new), "DLT not working"

@test_convergence
def camera_projection_levenberg_marquardt(A, real_points, screen_points):
    A_new = calibrate_camera.camera_projection_levenberg_marquardt(real_points, screen_points, A)

    assert np.allclose(A, A_new), "levenberg marquardt algorithm diverging from optimum"

    A_adj = A + np.random.rand(*A.shape)*1e-7

    A_new = calibrate_camera.camera_projection_levenberg_marquardt(real_points, screen_points, A_adj, iters=24)
    assert (np.abs(A - A_new) < 1e-5).all(), "levenberg marquardt does not converge to optimum from slightly different start"

@test_convergence
def numerical_camera_projection_levenberg_marquardt(A, real_points, screen_points):
    # TODO this test is super flaky with a random matrix and I have no idea why
    A_new = calibrate_camera.numerical_camera_projection_levenberg_marquardt(real_points, screen_points, A)

    assert np.allclose(A, A_new), "numerical levenberg marquardt algorithm diverging from optimum"

    assemble_func = calibrate_camera.assemble_feature_vector
    def test_assemble_feature_vector(P, include_K=True):
        a = assemble_func(P)
        return a + np.random.rand(*a.shape)*1e-6
    calibrate_camera.assemble_feature_vector = test_assemble_feature_vector

    A_new = calibrate_camera.numerical_camera_projection_levenberg_marquardt(real_points, screen_points, A, iters=15)

    assert (np.abs(A - A_new) < 1e-4).all(), "numerical levenberg marquardt does not converge to optimum from slightly different start"

@test_convergence
def calibrate_camera_test(A, real_points, screen_points):
    A_new, _ = calibrate_camera.calibrate_camera(real_points, screen_points)
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

    A_new, _ = calibrate_camera.calibrate_camera_const_internals(real_points, screen_points, K, A_norm)

    assert (np.abs(A - A_new) < 1e-4).all(), "Calibrate Camera Not Converging"

@test
def zhangs_method():
    # np.random.seed(10)
    points = (1/5)*np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    mg = np.array(np.meshgrid(points, points))
    X = np.reshape(mg,(2,len(points)*len(points)))
    X = np.vstack([X,np.zeros(len(points)*len(points))])
    X = np.vstack([X,np.ones(len(points)*len(points))])

    K = np.array([[-13.50441236, 0.2388132, 0.15245594],
                  [ -0.        , -13.4455227, 0.65115269],
                  [  0.        , 0.       , 1.        ]])

    for _ in range(5):
        try:
            x_imgs = []
            for i in range(10):
                A = np.random.rand(3,4)
                A[-1, -1] = 1
                _, R, p = li_utils.get_projection_product_matricies(A)

                P = li_utils.product_matricies_to_projection_matrix(K, R, p)

                x_e = calibrate_camera.camera_project_points_operation(P, X)
                x_imgs.append(x_e)
            
            K_est = calibrate_camera.zhangs_method(x_imgs, X)

            assert np.allclose(K, K_est), "zhangs method not working"
        except np.linalg.LinAlgError as e:
            if str(e) == "Matrix is not positive definite":
                print("   Matrix is not positive definite, not sure why this happens sometimes, retrying...")
            else:
                raise np.linalg.LinAlgError(e)

if __name__ == "__main__":
    run_tests()