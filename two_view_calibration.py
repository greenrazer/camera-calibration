import numpy as np

import li_utils

import calibrate_camera

def enforce_rank_2_fundamental(F):
    U, D, V = np.linalg.svd(F)

    # since D should be roughly equal to diag([D[0,0],D[1,1],eps]) were eps is close to zero
    # we want to enforce rank 2 by making eps = 0

    F = U @ np.diag([D[0],D[1],0]) @ V
    return F

def generate_A_matrix(pts1, pts2):
    A = []
    for uv_1, uv_2 in zip(pts1.T, pts2.T):
        A.append(np.kron(uv_1, uv_2))
    A = np.array(A)
    return A

def estimate_fundamental_matrix(screen_points_1, screen_points_2):
    # we need at least 8 corresponding points
    # given corresponding screen points
    # each row of A comes from the co-planar constraint
    # uv_1_n.T F uv_2_n = 0 for all points n
    # we can expand and then rewrite as
    # (uv_1 (kron) uv_2).T @ vec(F) = 0
    # which is then just a simple find the null space problem
    # since F = K_1 @ R_1 @ S_B @ R_2.T @ K_2_inv
    # the svd of F will give us 2 rotation matrixs and a scaling matrix
    # we want to enforce a rank 2 contraint on the scaling matrix
    # we want to edit the middle scale matrix to be diag([D[0,0],D[1,1],0]) forcing rank 2
    A = generate_A_matrix(screen_points_1, screen_points_2)

    f = li_utils.get_approx_null_space(A)

    F = li_utils.unvec(f, shape=(3,3))

    F = enforce_rank_2_fundamental(F)

    return F

def enforce_rank_2_essential(E):
    U, D, V = np.linalg.svd(E)

    # since D should be equal to diag([a,a,0]) we want to enforce rank 2
    # and since E is only defined up to scale, we can replace D with diag([1,1,0])
    E = U @ np.diag([1,1,0]) @ V

    return E

def estimate_essential_matrix(calibrated_screen_points_1, calibrated_screen_points_2):
    # we need at least 8 corresponding points
    # given corresponding screen points
    # each row of A comes from the co-planar constraint
    # uv_1_n.T F uv_2_n = 0 for all points n
    # we can expand and then rewrite as
    # (uv_1 (kron) uv_2).T @ vec(E) = 0
    # which is then just a simple find the null space problem
    # this is the same as the fundemental matrix, but it changes from here:
    # since we know E = R_1 @ S_b @ R_2.T
    # the svd of E will give us 2 rotation matrixs and a scaling matrix U, D, V
    # we want to enforce a rank 2 contraint on the scaling matrix
    # D should already be roughly equal to diag([D[0,0], D[1,1], eps]) where eps is really small
    # since its rank 2 and the scale doesn't matter
    # we want to edit the middle scale matrix to be diag([1,1,0]) forcing rank 2
    A = generate_A_matrix(calibrated_screen_points_1, calibrated_screen_points_2)

    e = li_utils.get_approx_null_space(A)

    E = li_utils.unvec(e, shape=(3,3))

    E = enforce_rank_2_essential(E)

    return E

def get_canidate_essential_product_matrcies(E):
    # we assume that the first rotation is the identity and the second rotation matrix is the relitive rotation, so
    # E = R_1 @ S_b @ R_2.T = I @ S_b @ R_12.T = S_b @ R_12.T
    # there are other ways to parameratize but this is the easiest
    # since we also know E = U @ diag([1,1,0]) @ V.T from constructing the essential matrix
    # we know
    # U.T @ U = I because its a rotation
    # Z = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]] which is skew symmetric
    # W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] which is a rotation
    # and Z and W have the property ZW = diag([1,1,0])
    # we can
    # E = U @ diag([1,1,0]) @ V.T =U @ Z @ W @ V.T = U @ Z @ U.T @ U @ W @ V.T
    # where U @ Z @ U.T is skew symmetric so it's S_b
    # and U @ W @ V.T is a rotation so its R_12.T
    # one small issue,
    # there are 4 pairs of matricies Z and W that multiply to diag([1,1,0])
    # diag([1,1,0]) = Z@W = -Z.T @ W = -Z @ W.T = Z.T @ W.T
    # therefore there are 4 solutions for the decomp of Z
    # So we have 2 solutions for R and 2 solutions for S_b
    # and 4 possible essential matricies:
    # S_a @ R_a.T = S_b @ R_b.T = -S_a @ R_b.T = -S_b @ R_a.T

    U, D, V = np.linalg.svd(E)

    # normalize U and V because det(U) and det(V) need to be positive for this to work
    # since U and V are rotations the det is either -1 or 1
    U = U/np.linalg.det(U)
    V = V/np.linalg.det(V)

    u3 = U[:,-1][..., None]

    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    S_a, S_b = U @ Z @ U.T, U @ Z.T @ U.T
    R_a, R_b = U @ W @ V, U @ W.T @ V

    return S_a, S_b, R_a, R_b, u3

def is_in_front_of_camera(P,x):
    # sign determines whether a normlaized point X is infront of behind camera P
    # with [M | p] @ X = x
    # X = (X, Y, Z, T)
    # x = (x,y,w)
    # full equation is 
    # M = <first 3 cols of P>
    # sign(det(M))*w/(T*norm(<last row of M>))
    # but we can simplify because we only care about sign
    # w*T*det(M)
    # if we assume X is normalized which means T is 1
    # so this just equals det(M) * w
    return  x[-1] * np.linalg.det(P[:, :3]) > 0

R_base = li_utils.I_3
p_base = np.array([0,0,0])[...,None]
P_base = np.hstack([R_base, p_base])

def get_projection_matrix_and_product_matricies_from_essential_matrix(E, calibrated_screen_point_1, calibrated_screen_point_2):
    kx1 = calibrated_screen_point_1[...,None]
    kx2 = calibrated_screen_point_2[...,None]

    S_a, S_b, R_a, R_b, u3 = get_canidate_essential_product_matrcies(E)

    prod_mats = [
        (u3, R_a),
        (-u3, R_a),
        (u3, R_b),
        (-u3, R_b)
    ]

    P_canidates = []

    for s, R in prod_mats:
        P_canidate = np.hstack([R, s])
        P_canidates.append(P_canidate)
    
    c = -1
    for i, P in enumerate(P_canidates):
        X = li_utils.triangulate_points(P_base, P, kx1, kx2)
        X = li_utils.to_homo_coords(X)
        # we'd normally need to do this:
        # x1 = P_base @ X
        # but, because our system has P_base as [the identity|0] 
        # x1 = X(without the last row)
        # we can just take the Z value of X 

        x2 = P @ X

        # to check if point is in front of the camera for P_base 
        # we just need to check if Z > 0, because M = the identity so det(M) = 1

        if X[2] > 0 and is_in_front_of_camera(P, x2):
            c=i

    if c > -1:    
        return P_canidates[c], prod_mats[c]
    else:
        raise RuntimeError("no possible solutions for projection matrix from essential matrix... I do not think this is possible, If you see this, hi :)")


def assemble_essential_feature_vector(E, calibrated_screen_point_1, calibrated_screen_point_2):
    # we want to get Rb and S_b from E
    _, P_prods = get_projection_matrix_and_product_matricies_from_essential_matrix(E, calibrated_screen_point_1, calibrated_screen_point_2)
    s_b, R_b = P_prods
    w = li_utils.rotation_matrix_to_angles(R_b)

    feature_vec = np.vstack([s_b, w[...,None]])

    return li_utils.vec(feature_vec)

POSITION_SLICE = slice(0,3)
ROTATION_ANGLE_SLICE = slice(3,6)

def disassemble_essential_feature_vector(feature_vec):
    s_b = feature_vec[POSITION_SLICE]
    R_b = li_utils.rotation_angles_to_matrix(feature_vec[ROTATION_ANGLE_SLICE])

    return s_b[...,None], R_b

def numerical_essential_matrix_levenberg_marquardt(E_start, calibrated_screen_points_1, calibrated_screen_points_2, iters=10):

    kx1 = calibrated_screen_points_1
    kx2 = calibrated_screen_points_2

    def func(inp):
        s_b, R_b = disassemble_essential_feature_vector(inp)
        P1 = P_base
        P2 = np.hstack([R_b, s_b])

        X_est = li_utils.triangulate_points(P1, P2, kx1, kx2)

        X_h = li_utils.to_homo_coords(X_est)

        r1 = calibrate_camera.camera_project_points_operation(P1, X_h) - kx1
        r2 = calibrate_camera.camera_project_points_operation(P2, X_h) - kx2

        r = np.hstack([li_utils.vec(r1), li_utils.vec(r2)])

        return r
    
    def cost_func(inp):
        r = func(inp)
        cost = r.T @ r
        return cost
    
    def jacobian_residual_func(inp):
        J = li_utils.numerical_jacobian(func, inp)
        return J, func(inp)

    inp_start = assemble_essential_feature_vector(E_start, kx1[:, 0], kx2[:, 0])
    inp = li_utils.generic_levenberg_marquardt(inp_start, cost_func, jacobian_residual_func, iters=iters)
    s_b, R_b = disassemble_essential_feature_vector(inp)

    S_b = li_utils.vector_3_to_skew_symmetric_matrix(li_utils.vec(s_b))

    E = S_b @ R_b

    return E

def assemble_fundamental_feature_vector(F):
    feature_vec = li_utils.vec(F)
    return feature_vec

def disassemble_fundamental_feature_vector(feature_vec):
    F = li_utils.unvec(feature_vec, shape=(3,3))
    return F

def create_relative_projection_matrix_from_fundamental_matrix(F):
    # the epipoles of Fe_a=0 F.Te_b'=0
    # so the epipoles are the null space of F
    # we only need e_b for this
    e_b = li_utils.get_approx_null_space(F.T)

    # now P2 is defined as [e_b_times@F | e_b]
    e_b_times = li_utils.vector_3_to_skew_symmetric_matrix(e_b)
    P2 = np.hstack([e_b_times @ F, e_b[...,None]])
    return P2


def numerical_fundamental_matrix_levenberg_marquardt(F_start, screen_points_1, screen_points_2, iters=10):
    # we are trying to minimize
    # P_1 = [I3|0]
    # P_2 = [e_b_times@F | e_b]
    # triangulate X_est from P_1, P_2 and our corresponding points
    # since P_1 @ X_est = <corresponding_points 1>
    # and   P_2 @ X_est = <corresponding points 2>
    # we wanna minimize 
    # the sum of squared errors between P_1 @ X_est and <corresponding_points 1>
    # added to the sum of squared errors between P_2 @ X_est and <corresponding_points 2>
    x1 = screen_points_1
    x2 = screen_points_2

    def func(inp):
        F = disassemble_fundamental_feature_vector(inp)
        P1 = P_base
        P2 = create_relative_projection_matrix_from_fundamental_matrix(F)

        X_est = li_utils.triangulate_points(P1, P2, x1, x2)

        X_h = li_utils.to_homo_coords(X_est)

        r1 = calibrate_camera.camera_project_points_operation(P1, X_h) - x1
        r2 = calibrate_camera.camera_project_points_operation(P2, X_h) - x2

        r = np.hstack([li_utils.vec(r1), li_utils.vec(r2)])

        return r
    
    def cost_func(inp):
        r = func(inp)
        cost = r.T @ r
        return cost
    
    def jacobian_residual_func(inp):
        J = li_utils.numerical_jacobian(func, inp)
        return J, func(inp)

    inp_start = assemble_fundamental_feature_vector(F_start)
    inp = li_utils.generic_levenberg_marquardt(inp_start, cost_func, jacobian_residual_func, iters=iters)
    F = disassemble_fundamental_feature_vector(inp)

    return F

def get_calibrated_camera_location_relative_to_first(calibrated_screen_points_1, calibrated_screen_points_2,iters=15):
    E = estimate_essential_matrix(calibrated_screen_points_1, calibrated_screen_points_2)
    E = numerical_essential_matrix_levenberg_marquardt(E, calibrated_screen_points_1, calibrated_screen_points_2, iters = iters)
    P2, _ = get_projection_matrix_and_product_matricies_from_essential_matrix(E,calibrated_screen_points_1[:,0], calibrated_screen_points_2[:,0])
    return P_base, P2

def get_uncalibrated_camera_location_relative_to_first(screen_points_1, screen_points_2):
    F = estimate_fundamental_matrix(screen_points_1, screen_points_2)
    F1 = numerical_fundamental_matrix_levenberg_marquardt(F, screen_points_1, screen_points_2, iters = 15)
    # not really sure if this works?
    # I dont think its easy to get our real K,R,p out of 
    # our projection matrices because lets say both cameras have the same K
    # so the second matrix will give us the Relative K(?), R, p
    # I dont know what relative internal parameters would even mean, pysically
    P2 = create_relative_projection_matrix_from_fundamental_matrix(F1)
    return P_base, P2

