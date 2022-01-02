import numpy as np

I_3 = np.identity(3)
I_4 = np.identity(4)
e_3 = np.array([0,0,1]).T
rotate3d_around_z_180 = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

rotate3d_around_y_180 = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

rotate3d_around_x_180 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
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

def dlt(real_points, screen_points, normalize_inp=True):
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

    # Svd time
    _, _, V = np.linalg.svd(M, full_matrices=False)

    # Last column of V, the vector that "points in the direction of the null space"
    p = V.T[:, -1]

    P = p.reshape((3,4))

    return P/P[-1,-1]

def unique_qr(M):
    R, T = np.linalg.qr(M)
    # This qr decomp doesn't enforce that the diagonals of the upper triangle matrix are positive 
    # so I have to do it myself
    # one issue with this is that the rotation matrix may have determinant -1 which means it's mirrored
    signs = 2 * (np.diag(T) >= 0) - 1
    R = R * signs[np.newaxis, :]
    T = T * signs[:, np.newaxis]
    return R, T

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

    # # We normalize K because it means we have to worry about fewer terms
    # # after converting from homogeneous coordinates to euclidean coordindates
    # # we get the same result
    K = K / K[-1,-1]

    # if the camera constant is positive we want to rotate around the Z axis
    # so that the projection plane is on the right spot.
    if K[0,0] >= 0:
        R = rotate3d_around_z_180 @ R
        K = K @ rotate3d_around_z_180

    return K, R, camera_pos[..., None]

def fix_rotation_matrix(K, R):
    if K[1,1] < 0:
        return K @ rotate3d_around_y_180, rotate3d_around_y_180 @ R
    return K, R

def product_matricies_to_projection_matrix(K, R, p):
    H = K@R
    camera_matrix = np.hstack((I_3, -p))
    P_new = H@camera_matrix
    return P_new

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

def point_lists_to_homogeneous_coordinates(real_points, screen_points):
    X = to_homo_coords(real_points.T)
    x = to_homo_coords(screen_points.T)
    return X, x

def linear_gauss_newton_method(real_points, screen_points, P_start, callback = None):
    # https://math.stackexchange.com/questions/4006567/how-jacobian-is-defined-for-the-function-of-a-matrix
    # https://math.stackexchange.com/questions/4309366/trouble-understanding-the-gauss-newton-method-to-update-a-matrix

    X, x = point_lists_to_homogeneous_coordinates(real_points, screen_points)

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

def generic_levenberg_marquardt(start, cost_func, jacobian_residual_func, iters=10, callback=None, learning_rate=1):
    # where lambda=0 does Gauss Newton and lambda >> 0 does gradient descent
    lambd = 1e-4
    curr = start
    min_cost = cost_func(start)

    for i in range(iters):
        J, r = jacobian_residual_func(curr)
        # The second derivative matrix is called the Hessian, it is linearly approximated by
        # H = J^T * J
        H = J.T @ J
        # The gauss newton method update term is H^-1 @ J^T @ r
        H_inv = np.linalg.pinv(H)
        # The gradient descent update term is diag(H) @ J^T @ r
        # we can switch between the 2 using the lambda term
        H_grad = lambd*np.diag(H)

        g = J.T @ r

        # so the final update term of the Levenberg Marquardt algorithm is
        # (H^-1 + lambda*diag(H)) @ J^T @ r
        update = (H_inv + H_grad) @ g

        # The learning rate is a number between 0 and 1 to ensure our steps
        # stay small as to not overshoot our goal, or end up somewhere we don't
        # want to be.
        # This is usually only important if our jacobian isn't very 
        # accurate (e.g. numerically estimated
        canidate = curr - learning_rate*update

        cost = cost_func(canidate)

        if cost < min_cost:
            curr = canidate
            min_cost = cost
            lambd /=10
            if callback is not None:
                callback(curr)
        else:
            lambd *= 10

        # If lambd ever gets this big, we've most likely already found the optimum
        # If not, we aren't likely to improve anymore before weird linalg errors start happening
        if lambd > 1e14:
            break
  
    if callback is not None:
        callback(curr)

    return curr

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

def intrinsic_camera_matrix_to_vector(K):
    # the intrinsic camera matrix in my program is defined as
    # [c, cs,     x_0]
    # [0, c(1+m), y_0]
    # [0, 0,    , 1  ]
    # the intrinsic vector should be [c, s, x_0, m, y_0]
    c = K[0,0]
    s = K[0,1]/c
    x_0 = K[0,2]
    m = K[1,1]/c - 1
    y_0 = K[1,2]
    return np.array([c, s, x_0, m, y_0])

def intrinsic_vector_to_camera_matrix(k):
    # given our intrinsic vector [c, s, x_0, m, y_0]
    # the intrinsic camera matrix in my program is defined as
    # [c, cs,     x_0]
    # [0, c(1+m), y_0]
    # [0, 0,    , 1  ]
    c, s, x_0, m, y_0 = k
    return np.array([
        [c, c*s,     x_0],
        [0, c*(1+m), y_0],
        [0, 0,       1  ],
    ])

def rotation_angles_to_matrix(w):
    # Rotation matrix R
    # convert w to w_times then
    # R = exp(w_times) ~  taylor expansion of e^{w_times}
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

    # if the rotation matrix is the identity we know the w value is 0,0,0
    if np.allclose(R, I_3):
        return np.array([0,0,0])

    t = np.trace(R)

    # If the trace of R is close to -1 theta will be close to infinity
    # which will mess with our calculations
    # therefore when theta is infinity we will return the last column of the
    # rotation matrix with 1 added to the last element
    if np.allclose(t, -1):
        o_p = 1+R[2,2]
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

