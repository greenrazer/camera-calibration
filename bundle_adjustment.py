import numpy as np

import li_utils
import sparse_li_utils

import calibrate_camera

CAMERA_ROTATION_ANGLE_SLICE = slice(0,3)
CAMERA_POSITION_SLICE = slice(3,6)

def dP_jacobian(feature_vec, K, X, gt):
    def func(curr):
        R = li_utils.rotation_angles_to_matrix(np.squeeze(feature_vec[CAMERA_ROTATION_ANGLE_SLICE]))
        p = np.squeeze(feature_vec[CAMERA_POSITION_SLICE])[...,None]
        P =  li_utils.product_matricies_to_projection_matrix(K, R, p)
        proj_points = calibrate_camera.camera_project_points_operation(P, X)
        # print(li_utils.to_euclid_coords(proj_points, entire=False) - gt)
        # #seems to be zero
        return li_utils.to_euclid_coords(proj_points, entire=False) - gt
    return li_utils.numerical_jacobian(func, feature_vec)

POSITION_SLICE = slice(0,3)

def dX_jacobian(feature_vec, P, gt):
    def func(curr):
        X = curr[...,None]
        homo_X = li_utils.to_homo_coords(X)
        proj_points = calibrate_camera.camera_project_points_operation(P, homo_X)
        # print(li_utils.to_euclid_coords(proj_points, entire=False) - gt)
        return li_utils.to_euclid_coords(proj_points, entire=False) - gt
    return li_utils.numerical_jacobian(func, feature_vec)

def make_jacobians(feature_vec, num_points, num_images, K, points2d, point_image_matrix):
    dX = {}
    dP = {}
    num_x_params = 3
    num_p_params = 6

    cameras, points3d = deconstruct_feature_vec(feature_vec, num_points, num_images, K, products=True)
    points3d = np.array(points3d)
    for i, camera in enumerate(cameras):
        # x, X = get_all_points_in_camera(i, points2d, points3d, point_image_matrix)
        # X_homo = li_utils.to_homo_coords(X)
        # x_homo = li_utils.to_homo_coords(x)
        R, p = camera
        P = li_utils.product_matricies_to_projection_matrix(K, R, p)

        w = li_utils.rotation_matrix_to_angles(R)
        p_vec = np.hstack([w, li_utils.vec(p)])

        for j in range(num_points):
            if point_image_matrix[i,j]:

                X_j = points3d.T[:, j]

                X_homo_j = li_utils.to_homo_coords(X_j[...,None])

                x_vec = li_utils.vec(X_j)

                lx = points2d[i,j,:][...,None]

                dP[(i, j)] = dP_jacobian(p_vec, K, X_homo_j, lx)
                dX[(i, j)] = dX_jacobian(x_vec, P, lx)

    return dX, dP

def generate_H_xx(num_points, num_images, dX, lambd, point_image_matrix):
    num_x_params = 3
    xs = num_points*num_x_params
    H_xx = sparse_li_utils.make_sparse_zeros(xs, xs)
    I = np.identity(num_x_params)
    for i in range(num_points):
        V_i = np.zeros((num_x_params, num_x_params))

        for j in range(num_images):
            if point_image_matrix[j, i]:
                J = dX[(j,i)]
                # print(J.T @ J)
                V_i += J.T @ J

        V_i += lambd*I
        # print(V_i)
        # quit()
        # print(V_i)
        at_x = i*num_x_params
        # print(at_x)
        # print(at_x+num_x_params)
        H_xx[at_x:at_x+num_x_params, at_x:at_x+num_x_params] = V_i
        # print(H_xx[at_x:at_x+num_x_params, at_x:at_x+num_x_params])
        # print()
        # np.set_printoptions(threshold=999999999999, linewidth=1000)
        # print((abs(H_xx.toarray()) > 0) *1)
        # print()
        # ss = sparse_li_utils.convert_to_csr_sparse_matrix_for_computation(H_xx)
        # print(ss[at_x:at_x+num_x_params, at_x:at_x+num_x_params])
    # print(H_xx)
    return sparse_li_utils.convert_to_csr_sparse_matrix_for_computation(H_xx)

def generate_H_pp(num_points, num_images, dP, lambd, point_image_matrix):
    num_p_params = 6
    ps = num_images*num_p_params
    H_pp = sparse_li_utils.make_sparse_zeros(ps, ps)
    I = np.identity(num_p_params)
    for j in range(num_images):
        U_j = np.zeros((num_p_params, num_p_params))
        for i in range(num_points):
            if point_image_matrix[j, i]:
                J = dP[(j,i)]
                U_j += J.T @ J
        U_j += lambd*I
        at_p = j*num_p_params
        H_pp[at_p:at_p+num_p_params, at_p:at_p+num_p_params] = U_j
        # print()
        # np.set_printoptions(threshold=999999999999, linewidth=1000)
        # print((abs(H_pp.toarray()) > 0) *1)
        # print()
        # quit()
    return sparse_li_utils.convert_to_csr_sparse_matrix_for_computation(H_pp)

def generate_H_xp(num_points, num_images, dX, dP, point_image_matrix):
    num_x_params = 3
    num_p_params = 6
    xs = num_points*num_x_params
    ps = num_images*num_p_params
    H_xp = sparse_li_utils.make_sparse_zeros(xs, ps)
    for i in range(num_points):
        for j in range(num_images):
            if point_image_matrix[j, i]:
                J_P = dP[(j,i)]
                J_X = dX[(j,i)]
                W_ij = J_X.T @ J_P
                at_x = i*num_x_params
                at_p = j*num_p_params
                H_xp[at_x:at_x+num_x_params, at_p:at_p+num_p_params] = W_ij
    return sparse_li_utils.convert_to_csr_sparse_matrix_for_computation(H_xp)

def generate_h_x(feature_vec, num_points, num_images, points, point_image_matrix, dX, K):
    num_p_params = 6
    num_x_params = 3
    xs = num_points*num_x_params
    h_x = np.zeros(xs)[..., None]
    for i in range(num_points):
        at_x = i*num_x_params
        x_vec = feature_vec[at_x:at_x+num_x_params]
        q_i = 0
        for j in range(num_images):
            if point_image_matrix[j,i]:
                at_p = j*num_p_params
                p_vec = feature_vec[xs+at_p:xs+at_p+num_p_params]

                J = dX[(j,i)]

                X = np.squeeze(x_vec[0:3])[...,None]
                R = li_utils.rotation_angles_to_matrix(np.squeeze(p_vec[0:3]))
                p = np.squeeze(p_vec[3:6])[...,None]
                P =  li_utils.product_matricies_to_projection_matrix(K, R, p)
                x = li_utils.to_homo_coords(points[j,i,:][...,None])

                homo_X = li_utils.to_homo_coords(X)

                r = x - calibrate_camera.camera_project_points_operation(P, homo_X)
                r = li_utils.cut_last_row(r)
                # print(J.T @ r)
                q_i += J.T @ r
                # print(q_i)
        h_x[at_x:at_x+num_x_params] = q_i
    return h_x

def generate_h_p(feature_vec, num_points, num_images, points, point_image_matrix, dP, K):
    num_p_params = 6
    num_x_params = 3
    ps = num_images*num_p_params
    xs = num_points*num_x_params
    h_p = np.zeros(ps)[..., None]
    for j in range(num_images):
        at_p = j*num_p_params
        p_vec = feature_vec[xs+at_p:xs+at_p+num_p_params]
        r_i = 0
        for i in range(num_points):
            if point_image_matrix[j, i]:
                at_x = i*num_x_params
                x_vec = feature_vec[at_x:at_x+num_x_params]

                J = dP[(j,i)]

                X = np.squeeze(x_vec[0:3])[...,None]
                R = li_utils.rotation_angles_to_matrix(np.squeeze(p_vec[0:3]))
                p = np.squeeze(p_vec[3:6])[...,None]
                P =  li_utils.product_matricies_to_projection_matrix(K, R, p)
                x = li_utils.to_homo_coords(points[j,i,:][...,None])

                homo_X = li_utils.to_homo_coords(X)

                r = x - calibrate_camera.camera_project_points_operation(P, homo_X)
                r = li_utils.cut_last_row(r)
                r_i += J.T @ r
        h_p[at_p:at_p+num_p_params] = r_i
    return h_p


def generate_bundle_adjustment_update(curr, points, point_image_matrix, lambd, K):
    # there are f number of images in the system
    # x_j = <k known nd points for image j> = nxk
    # X_j = <k unknown md points for image j> = mxk
    # P_j = <unknown nxm projection matrix for image j> 
    # P_j can be parametatized into some vector p_j, usually vec(P_j), but can be something else
    # v = <noise?>
    # if we make
    # params = [vec(X_0), vec(X_1), ..., vec(X_k),p_0, ..., p_f].T = [X_00, X_01, ..., X_fk,p_0, ..., p_f].T = [v_X|v_p].T
    # where
    # v_X = 1 x f*k*m
    # v_p = 1 x f*len(p_j)
    # so 
    # [v_X|v_p].T = (f*k*m + f*len(p_j)) x 1
    # then we can set up
    # A = [C|B] = k x (f*k*m + f*len(p_j))
    # such that 
    # C = k x f*k*m
    # B = k x len(p_j)*f
    # lets set up a gauss-newton equation
    # x_j + v = A*params = [C|B] * [v_x|v_p].T = C*v_x + C*v_p + B*v_x + B*v_p
    # but C as a function doesn't depend on v_p and B as a function doesnt depend on v_x so
    # A*params = C*v_x + 0 + 0 + B*v_p = C*v_x.T + B*v_p.T
    # now if we get slightly more specific with our A we can be even more clever
    # for an n x (m + len(p_j)) matrix A_ij, we only care about the md point i and the image j
    # x_ij + v_ij = A_ij * params = C_ij*v_x_i.T + B_ij*v_p_j.T
    # so A_ij = [0, ..., 0, C_ij, 0, ..., 0 |0, ..., 0, B_ij, 0, ..., 0]
    # where C_ij = n x m
    # and B_ij = n x len(p_j)
    # where i is the point index and j is the image index
    # we then create our coefficient matrix M
    # M = [A_00, A_10, A_20, ..., A_kf].T = [C, B]
    # M is super sparse, its mostly zeros
    # Sigma = <covariance of observations>
    # H = M.T * Sigma^-1 * M 
    #   = [[C^T * Sigma^-1 * C, C^T * Sigma^-1 * B],[B^T * Sigma^-1 * C, B^T * Sigma^-1 * B]] 
    #   = [[H_xx, H_xp],[H_px, H_pp]]
    # H_xx_i = diag(H_(x_i)(x_i))
    # H_pp_j = diag(H_(p_j)(p_j))
    # H_(x_i)(p_j) = C_ij Sigma_<ith point><jth camera>^-1 B_ij.T 
    # H_(p_j)(x_i) = H_(x_i)(p_i).T
    # H_(x_i)(x_i) = sum over all images i is observed( C_ij Sigma_<ith point><jth camera>^-1 C_ij.T )
    # H_(p_j)(p_j) = sum over all points in image j( B_ij Sigma_<ith point><jth camera>^-1 B_ij.T )
    # we can find C_ij and B_ij using the numerical jacobian
    # if Z = [[H_xx^-1, 0],[-H_px*H_xx^-1, I]]
    # then Z * H * [v_X|v_p].T = Z * h = h_bar
    # Z * H  = [[I, H_xx^-1 * H_xp],[0, H_pp - H_px*H_xx^-1*H_xp]] = [[I, Q],[0, H_pp_bar]]
    # if we solve just the bottom row of this equation we get
    # h_p_bar = h_p - H_px*H_xx^-1*h_x
    # H_pp_bar * v_p.T = h_p_bar
    # solve the above using a sparse solver
    # now we have v_p so we can find v_x via the first row
    # v_x.T + H_xx^-1 * H_xp * v_p.T = H_xx^-1 * h_x
    # v_x.T = H_xx^-1(h_x - H_xp * v_p.T)
    #
    # 1. compute H matrix parts using the above equations
    # 2. compute H_pp_bar and h_p_bar
    # 3. solve the sparse system H_pp_bar * v_p.T = h_p_bar
    # 4. solve v_x.T = H_xx^-1(h_x - H_xp * v_p.T)

    num_images = point_image_matrix.shape[0]
    num_points = point_image_matrix.shape[1]

    num_x_params = 3
    num_p_params = 6

    xs = num_points*num_x_params
    ps = num_images*num_p_params

    dX, dP = make_jacobians(curr, num_points, num_images, K, points, point_image_matrix)

    H_xx = generate_H_xx(num_points, num_images, dX, lambd, point_image_matrix)
    H_pp = generate_H_pp(num_points, num_images, dP, lambd, point_image_matrix)
    H_xp = generate_H_xp(num_points, num_images, dX, dP, point_image_matrix)
    H_px = H_xp.T

    H_xx_inv = sparse_li_utils.sparse_inv(H_xx)

    h_x = generate_h_x(curr, num_points, num_images, points, point_image_matrix, dX, K)
    h_p = generate_h_p(curr, num_points, num_images, points, point_image_matrix, dP, K)

    Y = H_px@H_xx_inv

    H_pp_bar = H_pp - Y@H_xp
    h_p_bar = h_p - Y@h_x

    v_p_T = sparse_li_utils.sparse_solve_linear_system(H_pp_bar, h_p_bar)[..., None]
    v_x_T = H_xx_inv@(h_x - (H_xp @ v_p_T))

    # np.set_printoptions(threshold=999999999999, linewidth=1000)
    # print((abs(H_pp.toarray()) > 0) *1)

    v = np.vstack([v_x_T, v_p_T])
    return v

def construct_feature_vec(num_points, num_images, K_R_Pos_tuple_list, Xs):
    start = np.zeros(num_points*3 + num_images*6)
    start[0:num_points*3] = li_utils.vec(Xs)
    for i, K_R_Pos_tuple in enumerate(K_R_Pos_tuple_list):
        _, R, p = K_R_Pos_tuple
        w = li_utils.rotation_matrix_to_angles(R)
        f = np.hstack([w, li_utils.vec(p)])
        start[num_points*3 + i*6 : num_points*3 + (i+1)*6] = f

    return start[...,None]

def deconstruct_feature_vec(feature_vec, num_points, num_images, K, products=False):
    num_x_params = 3
    num_p_params = 6

    xs = num_points*num_x_params

    cameras = []
    points = []
    for j in range(num_images):
        at_p = j*num_p_params
        p_vec = feature_vec[xs+at_p:xs+at_p+num_p_params]

        R = li_utils.rotation_angles_to_matrix(np.squeeze(p_vec[CAMERA_ROTATION_ANGLE_SLICE]))
        p = np.squeeze(p_vec[CAMERA_POSITION_SLICE])[...,None]
        if products:
            cameras.append((R, p))
        else:
            P = li_utils.product_matricies_to_projection_matrix(K, R, p)
            cameras.append(P)

    for i in range(num_points):
        at_x = i*num_x_params
        x_vec = feature_vec[at_x:at_x+num_x_params]
        X = x_vec
        points.append(li_utils.vec(X))

    return cameras, points

def get_all_points_in_camera(image_num, points2d, points3d, point_image_matrix):
    num_points = point_image_matrix.shape[0]

    points_2d = []
    points_3d = []
    for i in range(num_points):
        if point_image_matrix[i, image_num]:
            points_2d.append(points2d[image_num, i, :])
            points_3d.append(points3d[i,:])
    return np.array(points_2d).T, np.array(points_3d).T

def cost_func(feature_vec, K, points2d, point_image_matrix):
    num_x_params = 3
    num_p_params = 6

    num_points = point_image_matrix.shape[1]
    num_images = point_image_matrix.shape[0]

    cameras, points3d = deconstruct_feature_vec(feature_vec, num_points, num_images, K)
    cost = 0
    points3d = np.array(points3d)
    for i, camera in enumerate(cameras):
        x, X = get_all_points_in_camera(i, points2d, points3d, point_image_matrix)
        X_homo = li_utils.to_homo_coords(X)
        x_homo = li_utils.to_homo_coords(x)

        val = calibrate_camera.camera_projection_compute_cost(camera, X_homo, x_homo)
        cost += val * val
    
    return cost

def numerical_levenberg_marquardt_bundle_adjustment(start, points, point_image_matrix, K, iters=100):
    # Ideally we'd just do normal Lev-Mar optimisation on all the cameras and all the points
    # however, this linear system is gigantic, so we need to be clever
    # where lambda=0 does Gauss Newton and lambda >> 0 does gradient descent
    lambd = 1e-4
    curr = start
    min_cost = cost_func(start, K, points, point_image_matrix)

    for i in range(iters):
        update = generate_bundle_adjustment_update(curr, points, point_image_matrix, lambd, K)
        canidate = np.squeeze(curr)[...,None] + update
        cost = cost_func(canidate, K, points, point_image_matrix)
        if cost < min_cost:
            curr = canidate
            min_cost = cost
            lambd /=10
        else:
            lambd *= 10

        # If lambd ever gets this big, we've most likely already found the optimum
        # If not, we aren't likely to improve anymore before weird linalg errors start happening
        if lambd > 1e14:
            break

    cameras, points = deconstruct_feature_vec(curr, point_image_matrix.shape[1], point_image_matrix.shape[0], K)
    return cameras, points
