import numpy as np

I_3 = np.identity(3)
I_4 = np.identity(4)
e_3 = np.array([0,0,1]).T
rotate3d_around_z_180 = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

def construct_normalization_matrix(dim, avg, scale):
    trans_mat = np.zeros((dim, dim))
    trans_mat[0:dim-1, 0:dim-1] = np.identity(dim-1)
    trans_mat[0:dim-1, dim-1] = - avg
    trans_mat *= scale
    trans_mat[dim-1, dim-1] = 1
    return trans_mat

def normalize_points(pts):
    # list of points in an np array not a matrix 
    # column shape is the length of the dimenstions

    # we want the mean point to be the origin
    dims = pts.shape[1]
    avg = np.average(pts, axis=0)
    translated_pts = pts - avg

    # we want the average distance from the orgin to be sqrt(dimension)
    distances = np.linalg.norm(translated_pts, ord=dims, axis=1)
    avg_dist = np.average(distances)
    scale_factor = np.sqrt(dims)/avg_dist

    new_points = translated_pts * scale_factor

    return new_points, avg, scale_factor

def unnormalize_points(pts, avg, scale):
    return pts/scale + avg

def dlt(real_points, screen_points, normalize_inp=True, normalize_out=True):
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

    if normalize_inp:
        real, avg_real, scale_real = normalize_points(real_points)
        screen, avg_screen, scale_screen = normalize_points(screen_points)
    else:
        real = real_points
        screen = screen_points

    # OK step 1 build the M matrix
    M = []
    for xyz, uv in zip(real, screen):
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

    # Svd time
    _, _, V = np.linalg.svd(M, full_matrices=False)

    # Last column of V, the vector that "points in the direction of the null space"
    p = V.T[:, -1]

    P = p.reshape((3,4))

    if normalize_inp:
        real_norm_matrix   = construct_normalization_matrix(4, avg_real, scale_real)
        screen_norm_matrix = construct_normalization_matrix(3, avg_screen, scale_screen)
        screen_norm_matrix_inv = np.linalg.inv(screen_norm_matrix)
        P = screen_norm_matrix_inv@P@real_norm_matrix

    return P/P[-1,-1] if normalize_out else P

def get_projection_product_matricies(P):
    # now we have P = [H|h] H = KR, h = -KR<camera pos>
    # <camera pos> = - <H inverse>*h
    # QR decomposition turns a matrix into a triangular and a pure rotation matrix
    # we can do decomp on <H inverse> to get <R transpose> and <K inverse>
    # these are the not the normal constants? so we rotate the K matrix by 180 degrees around the z axis
    # and we normalize K by dividing it by K_(3_3)
    # we also rotate R by 180 around the z axis
    H = P[:, :3]
    # transformed position
    h = P[:, -1]

    H_inv = np.linalg.inv(H)

    camera_pos = - H_inv @ h

    R_T, K_inv = np.linalg.qr(H_inv)

    R = np.transpose(R_T)
    K = np.linalg.inv(K_inv)

    # We normalize K because it means we have to worry about fewer terms
    # after converting from homogeneous coordinates to euclidean coordindates
    # we get the same result
    K_norm = K/K[-1,-1]

    # Cause cameras have the "forward" direction facing away from the scene.
    # Our decomposition still holds because K * rotation_180 * rotation_180 * R = K*R
    # since rotation_180 * rotation_180 = I
    R = rotate3d_around_z_180 @ R
    K = K_norm @ rotate3d_around_z_180

    return K, R, camera_pos[..., None]

def cut_last_row(x):
    return x[0:-1, :]

def to_homo_coords(x):
    num = x.shape[1]
    row = np.ones(num)
    x = np.append(x,[row],axis=0)
    return x

def to_euclid_coords(x, entire=True):
    last_row = x[-1]
    X = x/last_row
    if entire:
        return X
    else:
        return cut_last_row(X)

def vec(M):
    return M.flatten('F')

def unvec(M, shape=(3,4)):
    return M.reshape(shape, order='F')

def dyadic_dot_product(A, B):
    return vec(A).T @ vec(B)

def linear_gauss_newton_method(real_points, screen_points, P_start, callback = None):
    # https://math.stackexchange.com/questions/4006567/how-jacobian-is-defined-for-the-function-of-a-matrix
    # https://math.stackexchange.com/questions/4309366/trouble-understanding-the-gauss-newton-method-to-update-a-matrix

    X = to_homo_coords(real_points.T)
    x = to_homo_coords(screen_points.T)

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

def khatri_rao_product(a,b):
    inside = [np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]
    return np.vstack(inside).T

def generate_permutation_matrix(size, swaps):
    K = np.identity(size)
    for s in swaps:
        K[:,[s[1],s[0]]] = K[:,[s[0],s[1]]]
    return K

swaps = [[2,8], [5,9], [8,10]]
swap_matrix = generate_permutation_matrix(12, swaps)
perm = np.kron(e_3.T, I_4) @ swap_matrix

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
    D = np.diag(V.T @ e_3)
    D_inv = np.linalg.inv(D)
    U = V @ D_inv

    T1 = np.kron(D_inv @ X.T, I_3)
    T2 = khatri_rao_product(D_inv, U) @ X.T @ perm

    J = T1 - T2
    R = U - x
    return J, R

def camera_projection_levenberg_marquardt_update(P_curr, X, x, lambd = 0):
    # The second derivative matrix is called the Hessian, it is linearly approximated by
    # H = J^T * J
    # The gauss newton method update term is H^-1 @ J^T @ r
    # The gradient descent update term is diag(H) @ J^T @ r
    # we can switch between the 2 using the lambda term
    # so the final update term of the Levenberg Marquardt algorithm is
    # (H^-1 + lambda*diag(H)) @ J^T @ r
    # where lambda=0 does Gauss Newton and lambda >> 0 does gradient descent

    J, R = camera_projection_levenberg_marquardt_jacobian_and_residual(P_curr,X, x)

    H = J.T @ J
    H_inv = np.linalg.inv(H)
    H_grad = lambd*np.diag(H)

    r = vec(R)

    g = J.T @ r

    update = (H_inv + H_grad) @ g

    return update

def camera_project_points_operation(P, X):
    V = P @ X
    D = np.diag(V.T @ e_3)
    D_inv = np.linalg.inv(D)
    return V @ D_inv

def camera_projection_compute_cost(P, X, x):
    U = camera_project_points_operation(P, X)
    R = U - x
    r = vec(R)
    return r.T @ r

def camera_projection_levenberg_marquardt(real_points, screen_points, P_start, iters= 10, callback = None):
    X = to_homo_coords(real_points.T)
    x = to_homo_coords(screen_points.T)

    lambd = 1e-3

    P_curr = P_start

    min_cost = camera_projection_compute_cost(P_curr, X, x)
    for i in range(iters):
        update = camera_projection_levenberg_marquardt_update(P_curr, X, x, lambd)

        p_canidate = vec(P_curr)
        p_canidate = p_canidate - update
        P_canidate = unvec(p_canidate)

        cost = camera_projection_compute_cost(P_canidate, X, x)

        if cost < min_cost:
            P_curr = P_canidate
            min_cost = cost
            lambd /=10
            if callback:
                callback(P_curr)
        else:
            lambd *= 10

        if lambd > 1e20:
            raise RuntimeError("Optimization diverging too rapidly. Try lowering the iterations.")
  
    if callback:
        callback(P_curr)

    return P_curr

def w_to_w_times(w):
    return np.array([
        [ 0,   -w[2], w[1]],
        [ w[2], 0,   -w[0]],
        [-w[1], w[0], 0]
    ])

def w_times_to_w(w_times):
    return np.array([
        w_times[2,1],
        w_times[0,2],
        w_times[1,0]
    ])

def intrinsic_camera_matrix_to_vector(K, full=True):
    # intrinsic vector in my program is defined as
    # k = [f_x, s, x_0, f_y, y_0]
    # we will assume f_x and f_y are the same, and s is 0
    # so the final output will be [f, x_0, y_0]
    if full:
        return np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2]])
    else:
        return np.array([K[0,0], K[0,2], K[1,2]])

def intrinsic_vector_to_camera_matrix(k, full=True):
    # normally
    # f = k[0] = f_x -> (0,0)
    # s = 0 -> (0,1)
    # k[1] = x_0 -> (0,2)
    # f = k[0] = f_y -> (1,1)
    # k[2] = y_0 -> (1,2)
    # 1 -> (2,2)
    # 0 everywhere else
    if full:
        return np.array([
            [k[0], k[1], k[3]],
            [0,    k[2], k[4]],
            [0,    0,    1   ]
        ])
    else:
        return np.array([
            [k[0], 0,    k[1]],
            [0,    k[0], k[2]],
            [0,    0,    1],
        ])


def rotation_angles_to_matrix(w):
    # Rotation matrix R
    # convert w to w_times then
    # R = exp(w_times) ~ first 3 terms of taylor expansion of e^{w_times}
    #                   = I + first_term*w_times + second_term*w_times^2
    # first_term = sin(|w|)/|w|
    # second_term = (1-cos(|w|))/|w|^2
    theta = np.linalg.norm(w)
    w_no_theta = w/theta

    t1 = np.sin(theta)
    t2 = (1-np.cos(theta))

    w_times = w_to_w_times(w_no_theta)
    return I_3 + t1*w_times + t2*w_times@w_times                        

def rotation_matrix_to_angles(R):
    # rotation angles w
    # R^-1 = R^T
    # w_times = log(R) = exp^-1(R) = inverse of above
    if np.allclose(R, I_3):
        return np.array([0,0,0])

    t = np.trace(R)

    if np.allclose(t, -1):
        o_p = 1+R[3,3]
        v = 1/np.sqrt(2*o_p)
        return v*np.array([R[0,2], R[1,2], o_p])

    theta = np.arccos((t-1)/2)
    t = 1/(2*np.sin(theta))
    w_times = t*(R-R.T)

    return theta*w_times_to_w(w_times)


class NDArrayIterator:
    def __init__(self, arr):
        self.max_counts = [i for i in arr.shape]

    def __count_increment(self):
        for i in range(len(self.count)):
            self.count[i] += 1
            if self.count[i] < self.max_counts[i]:
                return True
            self.count[i] -= self.max_counts[i]
        return False

    def __iter__(self):
        self.count = [0 for _ in self.max_counts]
        self.start = True
        return self

    def __next__(self):
        if self.start or self.__count_increment():
            self.start = False
            return tuple(self.count)
        raise StopIteration

def numerical_jacobian(f, inp):
    out_0 = f(inp)
    jacobian = np.zeros((*out_0.shape, *inp.shape))
    for index in NDArrayIterator(inp):
        epsilon = max(abs(inp[index]*1e-8), 1e-8)
        inp[index] += epsilon
        out_temp = f(inp)
        inp[index] -= epsilon

        dout_dinp = (out_temp - out_0)/epsilon

        jac_ind = tuple(slice(None, None, None) for i in range(len(out_0.shape))) + index
        jacobian[jac_ind] = dout_dinp

    size = (np.prod(out_0.shape), np.prod(inp.shape))
    return unvec(jacobian, shape=size)

def camera_project_points(P, real_points, raw_matrix=False):
    X = to_homo_coords(real_points.T)
    U = camera_project_points_operation(P, X)
    return U if raw_matrix else cut_last_row(U).T

def matrix_approx_equal(A, B, epsilon=1e-7, debug=False):
    diff = np.abs(A - B)
    bool_mat = diff > epsilon
    if debug:
        _, e = np.frexp(diff)
        print(e)
        print(bool_mat * 1)
    return not bool_mat.any()

def product_matricies_to_projection_matrix(K, R, p):
    H = K@R
    camera_matrix = np.hstack((I_3, -p))
    P_new = H@camera_matrix
    return P_new

def assemble_feature_vector(P):
    internal, rotation, position = get_projection_product_matricies(P)

    intrinsic = intrinsic_camera_matrix_to_vector(internal)
    w = rotation_matrix_to_angles(rotation)
    feature_vec = np.concatenate([intrinsic, w, vec(position)])

    return feature_vec

INTRINSIC_SLICE = slice(0,5)
ROTATION_ANGLE_SLICE = slice(5,8)
POSITION_SLICE = slice(8,11)

def disassemble_feature_vector(feature_vec):
    K = intrinsic_vector_to_camera_matrix(feature_vec[INTRINSIC_SLICE])
    R = rotation_angles_to_matrix(feature_vec[ROTATION_ANGLE_SLICE])
    p = feature_vec[POSITION_SLICE][...,None]

    P = product_matricies_to_projection_matrix(K, R, p)

    return P

def make_projection_function(X, x):
    def wrapper(inp):
        P = disassemble_feature_vector(inp)
        return camera_project_points_operation(P, X) - x
    return wrapper

def numerical_camera_projection_levenberg_marquardt(real_points, screen_points, P_start, iters=10, callback = None):
    X = to_homo_coords(real_points.T)
    x = to_homo_coords(screen_points.T)

    min_cost = camera_projection_compute_cost(P_start, X, x)

    curr_inp = assemble_feature_vector(P_start)
    func = make_projection_function(X, x)

    lambd = 1e-4

    for _ in range(iters):
        J = numerical_jacobian(func, curr_inp)

        H = J.T @ J
        H_inv = np.linalg.pinv(H)
        
        H_grad = lambd*np.diag(H)
        R = func(curr_inp)
        r = vec(R)
        g = J.T @ r

        update = (H_inv + H_grad) @ g

        curr_inp_canidate = curr_inp - update

        R = func(curr_inp_canidate)
        r = vec(R)
        cost = r.T @ r

        if cost < min_cost:
            curr_inp = curr_inp_canidate
            min_cost = cost
            lambd /= 10

            if callback:
                P = disassemble_feature_vector(curr_inp)
                callback(P)
        else:
            lambd *= 10

        if lambd > 1e15:
            raise RuntimeError("Optimization diverging too rapidly.")

    P = disassemble_feature_vector(curr_inp)

    if callback:
        callback(P)

    return P

def calibrate_camera(real_points, screen_points):
    P_init = dlt(real_points, screen_points)
    P = camera_projection_levenberg_marquardt(real_points, screen_points, P_init)
    internal, rotation, position = get_projection_product_matricies(P)
    return internal, rotation, position, P

def test_numerical_jacobian(real_points, screen_points, callb=None):
    X = to_homo_coords(real_points.T)
    x = to_homo_coords(screen_points.T)

    # def x(inp):
    #     return 3*inp

    # J = numerical_jacobian(x, np.array([[1.0,2.0],[3.0,4.0]]))
    # #I expect -> I_4*3.0 -> works

    # J = numerical_jacobian(lambda P:P@X, np.ones((3,4)))
    # #I expect -> X.T kron I_3 -> works

    K, R, pos, P = calibrate_camera(real_points, screen_points)

    # PP = product_matricies_to_projection_matrix(K, R, pos)
    # callb(PP, True)

    intrinsic = intrinsic_camera_matrix_to_vector(K, full=True)
    w = rotation_matrix_to_angles(R)
    # curr_inp = np.array([intrinsic, w, vec(pos)]).T
    curr_inp = np.concatenate([intrinsic, w, vec(pos)])
    print(curr_inp)
    INTRINSIC_SLICE = slice(0,5)
    ROTATION_ANGLE_SLICE = slice(5,8)
    POSITION_SLICE = slice(8,11)

    # curr_inp += np.random.randn(*curr_inp.shape)*0.3

    def func_to_proj(inp):
        K = intrinsic_vector_to_camera_matrix(inp[INTRINSIC_SLICE], full=True)
        R = rotation_angles_to_matrix(inp[ROTATION_ANGLE_SLICE])
        p = inp[POSITION_SLICE][...,None]


        # K = np.array([[-1.3,  0.  , 0.6],
        #                 [ 0.,   1. ,  0.6],
        #                 [ 0.,   0.,   1. ]])
        # p = np.array([[ 0. ],
        #                 [ 0.4],
        #                 [-0.7]])


        return product_matricies_to_projection_matrix(K, R, p)

    def func_V(P):
        V = P @ X
        return V

    def func_D_inv(V):
        D = np.diag(V.T @ e_3)
        D_inv = np.linalg.inv(D)
        return V @ D_inv

    def func_R(U):
        return U - x

    def func_all(inp):
        P = func_to_proj(inp)
        p1 = func_V(P)
        p2 = func_D_inv(p1)
        return func_R(p2)

    def cost(inp):
        R = func_all(inp)
        r = vec(R)
        return r.T @ r

    # from autograd import jacobian
    # J0 = jacobian(func_all)(curr_inp).reshape((21,9), order='F')
    # J1 = numerical_jacobian(func_all, curr_inp)
    # print(J0.shape, J1.shape, J2.shape)
    # print("0")
    # print(J0)
    # print(J1)
    # eq = matrix_approx_equal(J0, J1)

    # # print("1")
    # # print(J1)
    # # print("2")
    # # print(J2)

    # # return

    # J0 = jacobian(func_all)

    # k = 10
    # n = 3
    # m = 4
    # R = np.arange(n*k).reshape((n,k))
    # print(R)
    # r = R.flatten(order='F')[...,None]
    # print(r)

    # J = np.arange(n*k*n*m).reshape((n,k,n,m))
    # print(J[1,1, :, :])
    # print(J[1,2, :, :])
    # print(J[1,3, :, :])
    # print(J[1,4, :, :])
    # print(J[2,1, :, :])
    # print(J[2,2, :, :])
    # print(J[2,3, :, :])
    # print(J[2,4, :, :])
    # j0 = np.arange(n*k*n*m).reshape((n*m, n*k))
    # j1 = J.reshape((n*k, n*m))
    # print(j0)
    # # matrix_approx_equal(j0, j1)
    # return
    min_cost_diff = 1e-6
    cost_diff = min_cost_diff+1
    prev_cost = cost(curr_inp)

    count = 0

    lambd = 1e-7

    for _ in range(10):
        J = numerical_jacobian(func_all, curr_inp)
        # J2 = numerical_jacobian_1D(func_all, curr_inp)
        # print(abs(J- J2))
        # return
        # print(J)
        # print(J)
        # J = J0(curr_inp).reshape((21,9), order='F')
        # print("J")
        # print(J.shape)

        pos = curr_inp[POSITION_SLICE]
        print(pos)

        H = J.T @ J
        H_inv = np.linalg.pinv(H)
        
        H_grad = lambd*np.diag(H)
        R = func_all(curr_inp)
        r = vec(R)
        g = J.T @ r

        update = (H_inv + H_grad) @ g

        # print(lambd)

        curr_inp_canidate = curr_inp
        print("update:")
        print(update)
        curr_inp_canidate -= update
        # curr_inp_canidate = unvec(curr_inp_canidate, (3,3))

        R = func_all(curr_inp_canidate)
        r = vec(R)

        cost = r.T@r
        print('prev cost', prev_cost)
        print('cost', cost)

        if prev_cost is None or cost < prev_cost:
            print("hi2")
            curr_inp = curr_inp_canidate
            prev_cost = cost
            lambd /= 10


            intrinsic = curr_inp[INTRINSIC_SLICE]
            w = curr_inp[ROTATION_ANGLE_SLICE]
            pos = curr_inp[POSITION_SLICE]
            K = intrinsic_vector_to_camera_matrix(intrinsic, full=True)
            R = rotation_angles_to_matrix(w)
            import draw_utils
            # draw_utils.show_scene(real_points, K, R, pos, box_radius=3)
            PP = product_matricies_to_projection_matrix(K, R, pos[...,None])
            if callb:
                callb(PP)
        else:
            lambd *= 10

        if lambd > 1e15:
            raise RuntimeError("Optimization diverging too rapidly.")


        # prev_cost = cost
        # print(cost)

        count += 1
    
    intrinsic = curr_inp[INTRINSIC_SLICE]
    w = curr_inp[ROTATION_ANGLE_SLICE]
    pos = curr_inp[POSITION_SLICE]
    K = intrinsic_vector_to_camera_matrix(intrinsic, full=True)
    R = rotation_angles_to_matrix(w)

    print(pos)

    # import draw_utils
    # draw_utils.show_scene(real_points, K, R, pos, box_radius=3)
    PP = product_matricies_to_projection_matrix(K, R, pos[...,None])
    if callb:
        callb(PP, booll=True)

    return PP

    z = func_to_proj(curr_inp)
    p1 = func_V(z)
    p2 = func_D_inv(p1)
    return cut_last_row(p2).T










