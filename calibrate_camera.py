import numpy as np

import li_utils


def dlt(real_points, screen_points):
    # Given a list of real world points (e.g.? e.i? figure out later TODO x,y,z) 
    # and a list of correspoding screen points (x,y)
    #
    # So the real world points are 3d points in the real world and 
    # the screen points are the 2d point they map to on the screen 
    # so screen_point(2d) = <some projection matrix> * real_point(3d)
    # <some projection matrix> = <calibration matrix> * <rotation matrix> * <The 3d identity with a *negitive(not sure why important)* 3dvector represnting the camera location appended on the right>
    # <3 x 1> = <3 x 3> <3 x 3> <3 x 4>(made up of <3 x 3> with -<3 x 1> appended>) <4 x 1>
    # in non homogeneous coordinates(TODO??) ok so they're just used for easy rotations like affine transforms, ok not sure
    # the 4d vector is just <x y z 1> because to convert into euclidien space we divide by the last element
    # need more than 6 points because of degrees of freedom(?)
    # a <3 x 4> matrix P can be seperated into each of its rows <Prow1>(<4 x 1>), <Prow2>(<4 x 1>), <Prow3>(<4 x 1>)
    #
    #
    # so in the end we get the equations 
    # if i = (<Prow1 transposed> * real_point), j = (<Prow2 transposed> * real_point), k = (<Prow3 transposed> * real_point)
    # then
    # <screen_point_x> = i/k
    # <screen_point_y> = j/k
    #
    # - i + <screen_point_x>*k = - 1*i + 0*j + <screen_point_x>*k = 0
    # - j + <screen point_y>*k =   0*i - 1*j + <screen point_y>*k = 0
    #
    #  right so were gonna just make one big ol' vector containing all elements of P called p = [A, B, C] so a <12 x 1> vector
    # <x stuff transposed> * p = 0
    # <y stuff transposed> * p = 0
    # <x stuff transposed> = (-x, -y, -z, -1, 0, 0, 0, 0, <screen_point_x>*<real_point_x>, <screen_point_x>*<real_point_y>, <screen_point_x>*<real_point_z>, <screen_point_x>) 
    # <y stuff transposed> = (0, 0, 0, 0, -x, -y, -z, -1, <screen_point_y>*<real_point_x>, <screen_point_y>*<real_point_y>, <screen_point_y>*<real_point_z>, <screen_point_y>)
    # so for all n points we have a big matrix with <(x/y) stuff transposed> as rows
    # its a <2*n(2 because x and y are rows) x 12> matrix called M
    # so we can solve M(known)*p(unknown) = 0 and get our projection matrix
    # 
    # so the Svd turns a matrix into a 
    # <rotation in the left dimension>(S) * <squash into right dimension and scaling on each axis in right dimension>(V) * <rotation in right dimention>(D)
    # thats not how it gets its name thats just what i'm naming the matricies
    # for a <n x m> matrix it gets turned into <n x n> x <n x m> x <m x m>
    # we can use it to find the null space of a matrix when the scaling factor in the V matrix is zero (or near zero, why not)
    # so we get p as the column in D that corresponds to the lowest scaling factor 

    # OK step 1 build the M matrix
    M = []
    for xyz, uv in zip(real_points, screen_points):
        M.append([
            - xyz[0],
            - xyz[1],
            - xyz[2],
            - 1,
            0,
            0,
            0,
            0,
            uv[0]*xyz[0],
            uv[0]*xyz[1],
            uv[0]*xyz[2],
            uv[0],
        ])

        M.append([
            0,
            0,
            0,
            0,
            - xyz[0],
            - xyz[1],
            - xyz[2],
            - 1,
            uv[1]*xyz[0],
            uv[1]*xyz[1],
            uv[1]*xyz[2],
            uv[1],
        ])

    M = np.array(M)

    # where M @ p = 0
    p = li_utils.get_approx_null_space(M)

    P = np.reshape(p, (3,4))

    return P/P[-1,-1]

def linear_gauss_newton_method(real_points, screen_points, P_start, callback = None):
    # https://math.stackexchange.com/questions/4006567/how-jacobian-is-defined-for-the-function-of-a-matrix
    # https://math.stackexchange.com/questions/4309366/trouble-understanding-the-gauss-newton-method-to-update-a-matrix

    X, x = li_utils.point_lists_to_homogeneous_coordinates(real_points, screen_points)

    P_curr = P_start
    I_n = np.identity(x.shape[0])
    J = np.kron(X.T, I_n)
    H = J.T @ J
    H_inv = np.linalg.inv(H)
    for _ in range(10):
        R = P_curr @ X - x
        r = R.flatten('F')
        g = J.T @ r
        update = H_inv @ g

        p = P_curr.flatten('F')
        p_new = p - update

        P_curr = p_new.reshape((3, 4))
        if callback:
            callback(P_curr)
        
    return P_curr

swaps = [[2,8], [5,9], [8,10]]
swap_matrix = li_utils.generate_permutation_matrix(12, swaps)
perm = np.kron(li_utils.e_3.T, li_utils.I_4) @ swap_matrix

def camera_projection_levenberg_marquardt_jacobian_and_residual(P_curr, X, x):
    # https://math.stackexchange.com/questions/4316558/trouble-with-the-jacobian-for-a-camera-projection-matrix/4318435#4318435
    # L = vec(R)^Tvec(R)
    # R = U - x
    # U = V*D^-1
    # V = P*X
    # D = diag(V^T*e_3)
    #
    # dL/dP = dR/dP = dU/dP
    # dU/dP = dV/dP * D^-1 + V * dD^-1/dP
    # dV/dP = I_3 * dP/dP * X
    # dD^-1/dP = - D^-1 * dD/dP * D^-1
    # dD/dP =  X^T * (dP/dP)^T * e_3
    #
    # If we vectorize p = vec(P) and r = vec(R) we can compute the Jacobian.
    # (*) is the kroneker product 
    # (k) is the khatri-rao product which is just the column wise kroneker product
    #     e.g. if A = [a, b, c] and B = [d, e, f], A (k) B = [a (*) d, b (*) e, c (*) f] 
    # dr/dp = T_1 - T_2
    # T_1 = (X * D^-1)^T (*) I_3
    #     = D^-1^T * X^T (*) I_3
    #     = D^-1 * X^T (*) I_3
    # T_2 = - (U (k) D^-1) * X^T * (I_4^T (*) e_3)^T * K
    # where K is the permutation mnxmn matrix that permutes columns 9,10,11,12 to 3,6,9,12
    # we can then get the Jacobian J
    # J = T_1 + T_2
    V = P_curr @ X
    D = np.diag(V.T @ li_utils.e_3)
    D_inv = np.linalg.inv(D)
    U = V @ D_inv

    T1 = np.kron(D_inv @ X.T, li_utils.I_3)
    T2 = li_utils.khatri_rao_product(D_inv, U) @ X.T @ perm

    J = T1 - T2
    R = U - x
    return J, R

def camera_project_points_operation(P, X):
    V = P @ X
    D = np.diag(V.T @ li_utils.e_3)
    D_inv = np.linalg.pinv(D)
    return V @ D_inv

def camera_projection_compute_cost(P, X, x):
    U = camera_project_points_operation(P, X)
    R = U - x
    r = li_utils.vec(R)
    return r.T @ r

def camera_project_points(P, real_points, raw_matrix=False):
    X = li_utils.to_homo_coords(real_points.T)
    U = camera_project_points_operation(P, X)
    return U if raw_matrix else li_utils.cut_last_row(U).T

def camera_projection_levenberg_marquardt(real_points, screen_points, P_start, iters= 10, callback = None):
    X, x = li_utils.point_lists_to_homogeneous_coordinates(real_points, screen_points)

    def cost_func(p):
        P = li_utils.unvec(p)
        return camera_projection_compute_cost(P, X, x)
    
    def jacobian_residual_func(p):
        P = li_utils.unvec(p)
        J, R = camera_projection_levenberg_marquardt_jacobian_and_residual(P, X, x)
        return J, li_utils.vec(R)

    p_start = li_utils.vec(P_start)
    p = li_utils.generic_levenberg_marquardt(p_start, cost_func, jacobian_residual_func)

    P = li_utils.unvec(p)

    return P/P[-1,-1]


def assemble_feature_vector(P, include_K=True):
    K, R, p = li_utils.get_projection_product_matricies(P)
    w = li_utils.rotation_matrix_to_angles(R)
    p_T = li_utils.vec(p)
    if include_K:
        intrinsic = li_utils.intrinsic_camera_matrix_to_vector(K)
        feature_vec = np.concatenate([p_T, w, intrinsic])
    else:
        feature_vec = np.concatenate([p_T, w])
    return feature_vec

POSITION_SLICE = slice(0,3)
ROTATION_ANGLE_SLICE = slice(3,6)
INTRINSIC_SLICE= slice(6,11)

def disassemble_feature_vector(feature_vec, include_K=True):
    if include_K:
        K = li_utils.intrinsic_vector_to_camera_matrix(feature_vec[INTRINSIC_SLICE])
    else:
        K = None
    R = li_utils.rotation_angles_to_matrix(feature_vec[ROTATION_ANGLE_SLICE])
    p = feature_vec[POSITION_SLICE][...,None]

    return K, R, p

def numerical_camera_projection_levenberg_marquardt(real_points, screen_points, P_start, iters=10, callback = None, K_given=None):
    X, x = li_utils.point_lists_to_homogeneous_coordinates(real_points, screen_points)

    use_generated_K = K_given is None

    # if K is given, we return that K every time
    # otherwise we optimise internal parameters as well
    def disassemble_with_K(inp):
        K, R, p = disassemble_feature_vector(inp, include_K=use_generated_K)
        K = K if use_generated_K else K_given

        P = li_utils.product_matricies_to_projection_matrix(K, R, p)
        return P

    def cost_func(inp):
        P = disassemble_with_K(inp)
        cost = camera_projection_compute_cost(P, X, x)
        return cost 
    
    def func(inp):
        P = disassemble_with_K(inp)
        return camera_project_points_operation(P, X) - x

    def jacobian_residual_func(inp):
        J = li_utils.numerical_jacobian(func, inp)
        R = func(inp)
        return J, li_utils.vec(R)

    inp_start = assemble_feature_vector(P_start, include_K=use_generated_K)
    inp = li_utils.generic_levenberg_marquardt(inp_start, cost_func, jacobian_residual_func, learning_rate=1e-1)
    P = disassemble_with_K(inp)

    return P/P[-1,-1]

def projective_3_point(calibrated_real_points, calibrated_screen_points):
    # given calibrated screen points kx, and real points kX
    # we get the position and rotation of a camera 

    direction_vectors = li_utils.normalized(calibrated_screen_points, axis=1)

    alpha = np.arccos(direction_vectors[:,0], direction_vectors[:,1])
    beta = np.arccos(direction_vectors[:,1], direction_vectors[:,2])
    gamma = np.arccos(direction_vectors[:,2], direction_vectors[:,0])

    a = np.linalg.norm(calibrated_real_points[:,0] - calibrated_real_points[:,1])
    b = np.linalg.norm(calibrated_real_points[:,1] - calibrated_real_points[:,2])
    c = np.linalg.norm(calibrated_real_points[:,2] - calibrated_real_points[:,0])

    # law of cosines
    # substitue u,v
    # solve for u,v using 4th degree polynomial

    a_min_c_over_b = (a*a - c*c)/(b*b)
    b_min_c_over_b = (b*b - c*c)/(a*a)
    b_min_a_over_b = (a*a - b*b)/(c*c)

    a_plus_c_over_b = (a*a + c*c)/(b*b)
    b_plus_c_over_a = (b*b + c*c)/(a*a)
    a_plus_b_over_c = (a*a + b*b)/(c*c)

    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)
    cos_c = np.cos(gamma)

    A_4 = (a_min_c_over_b - 1)**2 - 4*c*c*(cos_a*cos_a)/(b*b)
    A_3 = 4*(a_min_c_over_b*(1- a_min_c_over_b)*cos_b - (1 - a_plus_c_over_b)*cos_a*cos_c + 2*c*c*cos_a*cos_a*cos_b/(b*b))
    #oh my godddddd
    A_2 = 2*(a_min_c_over_b*a_min_c_over_b - 1 + 2*a_min_c_over_b*a_min_c_over_b*cos_b*cos_b + 2*b_min_c_over_b*cos_a*cos_a - 4*a_plus_c_over_b*cos_a*cos_b*cos_c + 2*b_min_a_over_b*cos_c*cos_c)
    A_1 = 4*(-a_min_c_over_b*(1+a_min_c_over_b)*cos_b + 2*a*a*cos_c*cos_c*cos_b/(b*b) - (1-a_plus_c_over_b)*cos_a*cos_c)
    A_0 = (1+a_min_c_over_b)**2 - 4*a*a*cos_c*cos_c/(b*b)

    vs = np.roots([A_4, A_3, A_2, A_1, A_0])

    #TODO: fix arbitrary v
    v = vs[0]

    s_1_sq = b*b/(1-v*v-2*v*cos_b)
    s_3 = v*np.sqrt(s_1_sq)

    raise NotImplementedError("Haven't finished this p3p algo")

def zhangs_method_step(X, x):
    M = []
    for xyz, uv in zip(X.T, x.T):
        M.append([
            - xyz[0],
            - xyz[1],
            - 1,
            0,
            0,
            0,
            uv[0]*xyz[0],
            uv[0]*xyz[1],
            uv[0],
        ])

        M.append([
            0,
            0,
            0,
            - xyz[0],
            - xyz[1],
            - 1,
            uv[1]*xyz[0],
            uv[1]*xyz[1],
            uv[1],
        ])

    M = np.array(M)

    h = li_utils.get_approx_null_space(M)

    # H = h.reshape((3,3))
    H = li_utils.unvec(h, shape=(3,3))

    def v_i_j(i,j):
        return np.array([
            [H[i,0]*H[j,0]],
            [H[i,0]*H[j,1] + H[i,1]*H[j,0]],
            [H[i,2]*H[j,0] + H[i,0]*H[j,2]],
            [H[i,1]*H[j,1]],
            [H[i,2]*H[j,1] + H[i,1]*H[j,2]],
            [H[i,2]*H[j,2]],
        ])

        # return np.array([
        #     [H[0,i]*H[0,j]],
        #     [H[0,i]*H[1,j] + H[1,i]*H[0,j]],
        #     [H[2,i]*H[0,j] + H[0,i]*H[2,j]],
        #     [H[1,i]*H[1,j]],
        #     [H[2,i]*H[1,j] + H[1,i]*H[2,j]],
        #     [H[2,i]*H[2,j]],
        # ])

    return v_i_j(0,1).T, v_i_j(0,0).T - v_i_j(1,1).T

def from_b_to_B(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = B[1,0] = b[1]
    B[0,2] = B[2,0] = b[2]
    B[1,1] = b[3]
    B[1,2] = B[2,1] = b[4]
    B[2,2] = b[5]
    return B

def zhangs_method(images, real_points):
    V = np.zeros((2*len(images), 6))
    for i, screen_points in enumerate(images):
        r1, r2 = zhangs_method_step(real_points, screen_points)
        V[2*i,:] = r1
        V[2*i+1,:] = r2
    b = li_utils.get_approx_null_space(V)

    B = from_b_to_B(b)

    K_inv_T = np.linalg.cholesky(B)

    K = np.linalg.inv(K_inv_T.T)

    K_norm = K / K[-1,-1]
    return K_norm @ li_utils.rotate3d_around_z_180


def calibrate_camera_helper(real_points, screen_points, func):
    real, avg_real, scale_real = li_utils.normalize_points(real_points)
    screen, avg_screen, scale_screen = li_utils.normalize_points(screen_points)

    P = func(real, screen)

    real_norm_matrix  = li_utils.construct_normalization_matrix(4, avg_real, scale_real)
    screen_norm_matrix = li_utils.construct_normalization_matrix(3, avg_screen, scale_screen)
    screen_norm_matrix_inv = np.linalg.inv(screen_norm_matrix)
    P_unnormalized = screen_norm_matrix_inv@P@real_norm_matrix
    P_unnormalized /= P_unnormalized[-1,-1]

    return P_unnormalized, P

def calibrate_camera_just_dlt(real_points, screen_points):
    def func(real, screen):
        P = dlt(real, screen)
        return P

    P_unnormalized, P = calibrate_camera_helper(real_points, screen_points, func)
    return P_unnormalized, P

def calibrate_camera(real_points, screen_points):

    def func(real, screen):
        P = dlt(real, screen)
        P = camera_projection_levenberg_marquardt(real, screen, P)
        P = numerical_camera_projection_levenberg_marquardt(real, screen, P)
        return P

    P_unnormalized, P = calibrate_camera_helper(real_points, screen_points, func)
    return P_unnormalized, P

# P_estimate and K should both be normalized.
def calibrate_camera_const_internals(real_points, screen_points, K, P_estimate):

    def func(real, screen):
        P = numerical_camera_projection_levenberg_marquardt(real, screen, P_estimate, K_given=K)
        return P

    P_unnormalized, P = calibrate_camera_helper(real_points, screen_points, func)
    return P_unnormalized, P