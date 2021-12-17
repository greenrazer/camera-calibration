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

def test_calibration(func):
    def wrapper():
        print(f"Starting test \"{func.__name__}\"...")
        passed = True
        try:
            temp_passed = func(real_points_m, points1)
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

    P_before = li_utils.dlt(real_points, screen_points)
    estimated_screen_points = li_utils.camera_project_points(P_before, real_points)
    P_after = li_utils.dlt(real_points, estimated_screen_points)

    return li_utils.matrix_approx_equal(P_before, P_after, debug=True)

if __name__ == '__main__':
    for test in tests:
        test()