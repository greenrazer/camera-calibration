import numpy as np

import li_utils

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
    A = []
    for uv_1, uv_2 in zip(screen_points_1, screen_points_2):
        A.append(np.kron(uv_1, uv_2).T)
    A = np.array(A)

    f = li_utils.get_null_space(A)

    F = li_utils.unvec(f, shape=(3,3))

    U, D, V = np.linalg.svd(F, full_matrices=False)

    # since D should be roughly equal to diag([D[0,0],D[1,1],eps]) were eps is close to zero
    # we want to enforce rank 2 by making eps = 0
    F = U @ np.diag([D[0],D[1],0]) @ V.T

    return F

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
    A = []
    for uv_1, uv_2 in zip(calibrated_screen_points_1.T, calibrated_screen_points_2.T):
        A.append(np.kron(uv_1, uv_2))
    A = np.array(A)

    e = li_utils.get_approx_null_space(A)

    E = li_utils.unvec(e, shape=(3,3))

    U, D, V = np.linalg.svd(E)

    # since D should be equal to diag([a,a,0]) we want to enforce rank 2
    # and since E is only defined up to scale, we can replace D with diag([1,1,0])
    E = U @ np.diag([1,1,0]) @ V

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

def get_canidate_projection_matricies_from_essential_matrix(E):
    S_a, S_b, R_a, R_b, u3 = get_canidate_essential_product_matrcies(E)

    # create 4 canidate projection matricies
    P_a = np.hstack([R_a, u3])
    P_b = np.hstack([R_a, -u3])
    P_c = np.hstack([R_b, u3])
    P_d = np.hstack([R_b, -u3])

    return P_a, P_b, P_c, P_d

def is_infront_of_camera(P,X,x):
    # sign determines whether a normlaized point X is infront of behind camera P
    # with [M | p] @ X = x
    # X = (X, Y, Z, T)
    # x = (x,y,w)
    # full equation is 
    # M = <first 3 cols of P>
    # sign(det(M))*w/(T*norm(<last row of M>))
    # but we can simplify because we only care about sign
    # w*T*det(M)
    # if we assume X is normalized which means T is1
    # so this just equals det(M) * w
    return  X[-1] * x[-1] * np.linalg.det(P[:, :3]) > 0

R_base = li_utils.I_3
p_base = np.array([0,0,0])[...,None]
P_base = np.hstack([R_base, p_base])

def get_projection_matricies_from_essential_matrix(E, calibrated_screen_point_1, calibrated_screen_point_2):
    P_a, P_b, P_c, P_d = get_canidate_projection_matricies_from_essential_matrix(E)
    P_canidates = [P_a, P_b, P_c, P_d]
    

    c = -1
    for i, P in enumerate(P_canidates):
        X = li_utils.triangulate_points(P_base, P, calibrated_screen_point_1[...,None], calibrated_screen_point_2[...,None])
        X = li_utils.to_homo_coords(X)
        x1 = P_base @ X
        x2 = P @ X

        if is_infront_of_camera(P_base, X, x1) and is_infront_of_camera(P, X, x2):
            c=i

    if c > -1:    
        return P_base, P_canidates[c]
    else:
        raise RuntimeError("no possibe solutions")

def get_uncalibrated_camera_location_relative_to_first(calibrated_screen_points_1, calibrated_screen_points_2):
    E = estimate_essential_matrix(calibrated_screen_points_1, calibrated_screen_points_2)
    P1, P2 = get_projection_matricies_from_essential_matrix(E,calibrated_screen_points_1.T[0], calibrated_screen_points_2.T[0])
    return P1, P2

