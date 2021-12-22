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
        except Exception as e:
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

@test_calibration
def camera_matrix_decomposition(real_points, screen_points):
    P_before = li_utils.dlt(real_points, screen_points)
    K, R, pos = li_utils.get_projection_product_matricies(P_before)
    P_after = li_utils.product_matricies_to_projection_matrix(K, R, pos)

    xt = np.random.rand(4,100)
    U_before = li_utils.camera_project_points_operation(P_before, xt)
    U_after = li_utils.camera_project_points_operation(P_after, xt)

    return li_utils.matrix_approx_equal(U_before, U_after)

@test_calibration
def direct_linear_transform(real_points, screen_points):
    # If P is the best estimation of the projection matrix using the dlt
    # If we perform the dlt again using the estimated screen points the error
    # should be close to zero and the new estimated projection should be roughly equal

    A = np.random.rand(3,4)
    A /= np.linalg.norm(A)
    A /= A[-1, -1]
    
    x = np.random.rand(3,12)
    x = li_utils.to_homo_coords(x)

    X_inp = li_utils.to_euclid_coords(x,   entire=False).T
    X_out = li_utils.to_euclid_coords(A@x, entire=False).T

    A_new = li_utils.dlt(X_inp, X_out)

    if np.allclose(li_utils.dyadic_dot_product(A, A_new), -1):
        A_new = -A_new

    assert np.allclose(A, A_new), "DLT not working"

    norm_real, avg_real, scale_real = li_utils.normalize_points(real_points)
    norm_screen, avg_screen, scale_screen = li_utils.normalize_points(screen_points)

    P_before = li_utils.dlt(norm_real, norm_screen)

    estimated_screen_points = li_utils.camera_project_points(P_before, real_points)
    P_after = li_utils.dlt(real_points, estimated_screen_points)

    return li_utils.matrix_approx_equal(P_before, P_after), "DLT not working"

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

@test
def axis_angle():
    w1 = (np.random.rand(3)-0.5)*2*np.pi

    while np.linalg.norm(w1) > np.pi:
        w1 = (np.random.rand(3)-0.5)*2*np.pi

    R1 = li_utils.rotation_angles_to_matrix(w1)
    w1_again = li_utils.rotation_matrix_to_angles(R1)

    assert np.allclose(w1, w1_again), "Applying inverse to matrix not working."

if __name__ == '__main__':
    for test in tests:
        test()