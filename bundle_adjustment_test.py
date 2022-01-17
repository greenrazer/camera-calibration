from test import test, run_tests

import numpy as np

import li_utils

import bundle_adjustment

import calibrate_camera

import two_view_calibration

@test
def bundle_adjustment_all_points_in_all_cams():
    # np.random.seed(40)

    # generate ground truth cameras 
    num_cams = 10
    true_cameras = []
    true_camera_products = []
    rand_camera_products = []
    rand_cameras=[]

    noise_sc = 0.5

    K = np.identity(3)
    for i in range(num_cams):
        p = np.random.rand(3)[...,None]
        pr = p + np.random.rand(3)[...,None]*noise_sc

        # K = np.diag(np.random.rand(3))
        # K /= K[-1,-1]

        w = np.random.rand(3)*0.1
        R =li_utils.rotation_angles_to_matrix(w)
        Rr =li_utils.rotation_angles_to_matrix(w + np.random.rand(3)*noise_sc)

        P = li_utils.product_matricies_to_projection_matrix(K, R, p)
        Pr = li_utils.product_matricies_to_projection_matrix(K, Rr, pr)
        true_cameras.append(P)
        rand_cameras.append(Pr)
        true_camera_products.append((K, R, p))
        rand_camera_products.append((K, Rr, pr))

    # generate ground truth 3d points

    num_points = 15
    true_points_3d = np.random.rand(3, num_points)
    true_points_3d += np.array([-1, -1, 2])[...,None]
    true_points_3d = li_utils.to_homo_coords(true_points_3d)

    # get 2d points from all cameras and the 3d points
    points_2d = []
    for true_cam in true_cameras:
        x = true_cam @ true_points_3d
        x = li_utils.to_euclid_coords(x, entire=False)
        points_2d.append(x.T)
    
    points_2d = np.array(points_2d)

    X_est = true_points_3d[:3, :] + np.random.rand(*true_points_3d[:3, :].shape)*noise_sc

    point_image_matrix = np.ones((num_cams, num_points))

    start = bundle_adjustment.construct_feature_vec(num_points, num_cams, rand_camera_products, X_est[:3,:])

    cameras, points = bundle_adjustment.numerical_levenberg_marquardt_bundle_adjustment(start, points_2d, point_image_matrix, K, iters=20)

    scale, R_sc, T = li_utils.iterative_closest_point_with_scale(true_points_3d[:3, :], np.array(points).T)

    sc_pts = (scale*R_sc@(np.array(points).T) + T).T 
    assert np.allclose(true_points_3d[:3, :].T, sc_pts), "real world cordinates incorrectly predicted"

    for i, c in enumerate(cameras):
        _,R,p = li_utils.get_projection_product_matricies(c)
        p = scale*R_sc@p + T
        R = li_utils.rotate3d_around_z_180 @ R @ R_sc.T
        _, R_true, p_true = true_camera_products[i]
        assert np.allclose(p_true, p), "camera position incorrect"
        assert np.allclose(R_true, R), "camera rotation incorrect"

    # def pos_rot_scene(pts, cameras, sc=True):
    #     positions = []
    #     rotations = []
    #     for c in cameras:
    #         _,R,p = li_utils.get_projection_product_matricies(c)
    #         if sc:
    #             p = scale*R_sc@p + T
    #             R = R_sc@R

    #         positions.append(p)
    #         rotations.append(R)

    #     if sc:
    #        pts = sc_pts

    #     import draw_utils
    #     draw_utils.show_multi_cam_scene(pts, positions, rotations)

    # pos_rot_scene(true_points_3d.T, true_cameras, sc=False)
    # # pos_rot_scene(X_est.T, rand_cameras)
    # pos_rot_scene(np.array(points), cameras)

@test
def deconstruct_feature_vec():
    num_points = 10
    num_cams = 10

    K = np.identity(3)

    Ps = []
    R_ps = []
    for c in range(num_cams):
        w = np.random.rand(3)
        R = li_utils.rotation_angles_to_matrix(w)
        p = np.random.rand(3)[...,None]
        R_ps.append((K, R, p))
        Ps.append(li_utils.product_matricies_to_projection_matrix(K, R, p))

    X_est = np.random.rand(3,num_points)

    start = bundle_adjustment.construct_feature_vec(num_points, num_cams, R_ps, X_est)

    cameras_dec, points = bundle_adjustment.deconstruct_feature_vec(start, num_points, num_cams, K)

    for cam_true, cam_dec in zip(Ps, cameras_dec):
        assert np.allclose(cam_true, cam_dec), "deconstruct feature vector not working"

    for x_true, x_dec in zip(X_est.T, points):
        assert np.allclose(x_true[...,None], x_dec[...,None]), "deconstruct feature vector not working"

if __name__ == '__main__':
    run_tests()
