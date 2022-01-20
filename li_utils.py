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

def normalize_points_for_each_image(pts):
    pts_out = []
    avgs = []
    scales = []
    for points in pts:
        new_pts, avg, scale = normalize_points(points)
        pts_out.append(new_pts)
        avgs.append(avg)
        scales.append(scale)
    return np.array(pts_out), np.array(avgs), np.array(scales)

def calibrate_homo_points_for_each_image(pts, K):
    K_inv = np.linalg.inv(K)
    pts_out = []
    for points in pts:
        cal_points = K_inv@points.T
        pts_out.append(cal_points.T)
    return np.array(pts_out)

def convert_to_euclid_for_each_image(pts):
    pts_out = []
    for points in pts:
        eu_points = to_euclid_coords(points.T, entire=False)
        pts_out.append(eu_points.T)
    return np.array(pts_out)

def convert_to_homo_for_each_image(pts):
    pts_out = []
    for points in pts:
        homo_points = to_homo_coords(points.T)
        pts_out.append(homo_points.T)
    return np.array(pts_out)

def get_approx_null_space(A):
    # Svd time
    _, _, V = np.linalg.svd(A, full_matrices=False)

    # Last column of V, the vector that "points in the direction of the null space"
    return V.T[:, -1]

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

def get_rotation_and_position_from_calibrated_projection_matrix(P):
    K, R, p = get_projection_product_matricies(P)

    if not np.allclose(I_3 @ rotate3d_around_z_180, K):
        print("Warning:")
        print("given matrix composed of more than just rotation and position")
        print("this most likely means there is a lot of error in the calibration matrix")
        print("used to normalize the data")

    # because our get_projection_product_matricies returns P = K @ rotate_180_z @ rotate_180_z @ R [I3|-p]
    # when we multiply by (K @ rotate_180_z)^-1 which is equal to rotate_180_z^-1 @ K^-1
    # rotate_180_z^-1 @ K^-1 @ P = rotate_180_z^-1 @ K^-1 @ K @ rotate_180_z @ rotate_180_z @ R [I3|-p]
    #                            = rotate_180_z @ R [I3|-p]
    # so there is a hanging 180 degree rotation we must get rid of
    return rotate3d_around_z_180 @ R, p

def product_matricies_to_projection_matrix(K, R, p):
    H = K@R
    camera_matrix = np.hstack((I_3, -p))
    P_new = H@camera_matrix
    return P_new if P_new[-1,-1] == 0 else P_new/P_new[-1,-1]

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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def vector_3_to_skew_symmetric_matrix(v):
    return np.array([
        [ 0,   -v[2], v[1]],
        [ v[2], 0,   -v[0]],
        [-v[1], v[0],  0]
    ])

def skew_symmetric_matrix_to_vector_3(v_times):
    return np.array([
        v_times[2,1],
        v_times[0,2],
        v_times[1,0]
    ])

#TODO delete
def w_to_w_times(w):
    return vector_3_to_skew_symmetric_matrix(w)

def w_times_to_w(w_times):
    return skew_symmetric_matrix_to_vector_3(w_times)

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
    if theta == 0:
        return I_3
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

def triangulate_point(P_1, P_2, projected_point_1, projected_point_2):
    # this method is pretty brittle... basically only works for stereo normal case
    # (e.g both cameras pointing in the same direction, normal to the line connecting their centers)
    # assuming we have
    # r = <vector from camera_center_1 to projected_point_1>
    # s = <vector from camera_center_2 to projected_point_2>
    # f(lambda) = <camera_center_1> + lambda*r
    # g(mu) = <camera_center_2> + mu*s
    # find lambda and mu such that we minimize the length of the orthoganal line connecting f(lambda) and g(mu)
    # we know the line f(lambda) - g(mu) must be orthoganal to the vector r and the vector s so
    # (f(lambda) - g(mu)).T @ r = 0 
    # (f(lambda) - g(mu)).T @ s = 0
    # expanding out we can turn this into a series of linear equations to get
    # A*[lambda, mu].T = b
    # where b = [d.T @ r, d.T @ s].T is a 2 x 1 matrix
    # d = vector between 2 centers = <camera_center_2> - <camera_center_1> is an 3 x 1 matrix
    # then we can get f and g using lambda and mu
    # the point we want is the average on the line between f and g or f+g/2

    R_1 = P_1[:, :3]
    R_2 = P_2[:, :3]

    p_1 = P_1[:,-1][...,None]
    p_2 = P_2[:,-1][...,None]

    r = R_1.T @ projected_point_1
    s = R_2.T @ projected_point_2

    A = np.array([
        [r.T @ r, -s.T @ r],
        [r.T @ s, -s.T @ s],
    ])
    A = np.squeeze(A)

    d = p_2 - p_1
    b = np.array([
        [d.T @ r],
        [d.T @ s]
    ])
    b = np.squeeze(b)

    # solve Ax = b where x = [lambda, mu].T 
    x = np.linalg.solve(A, b)
    lam = x[0]
    mu = x[1]
    f = p_1 + lam*r
    g = p_2 + mu*s

    # average of f and g
    return (f+g) / 2

def triangulate_points(P1, P2, projected_points_1, projected_points_2):
    num_points = projected_points_1.shape[1]
    X_est = np.zeros((4,num_points))
    for i in range(num_points):
        x1, y1 =  projected_points_1[0:2, i]
        x2, y2 =  projected_points_2[0:2, i]

        A = np.array([
            x1*P1[2] - P1[0],
            y1*P1[2] - P1[1],
            x2*P2[2] - P2[0],
            y2*P2[2] - P2[1],
        ])

        U, S, V = np.linalg.svd(A)
        X = V.T[:,-1]
        X_est[:,i] = X
    
    return to_euclid_coords(X_est, entire=False)

def compute_self_covariance_for_pts(X):
    cov = 0
    num_points = X.shape[1]
    for i in range(num_points):
        x = X[:,i][...,None]
        cov += np.squeeze(x.T @ x)
    return cov/num_points

def iterative_closest_point_with_scale(control_points, corresponding_points):
    # finds lambda, R, and T such that
    # control_points = lambda*R @ corresponding_points + T

    mean_control = np.mean(control_points, axis=1)[...,None]
    mean_corresponding = np.mean(corresponding_points, axis=1)[...,None]

    translated_control = control_points - mean_control
    translated_corresponding = corresponding_points - mean_corresponding

    control_cov = compute_self_covariance_for_pts(translated_control)
    corresponding_cov = compute_self_covariance_for_pts(translated_corresponding)

    lambd_sq = control_cov/corresponding_cov
    lambd = np.sqrt(lambd_sq)

    scale = np.sqrt(lambd)

    scaled_control = (1/scale) * translated_control
    scaled_corresponding = scale * translated_corresponding

    H = np.zeros((control_points.shape[0], control_points.shape[0]))
    for col in range(control_points.shape[1]):
        H += scaled_corresponding[:, col][...,None] @ scaled_control[:, col][None,...]
    
    U, D, V = np.linalg.svd(H)
    R = V.T @ U.T

    T = mean_control- lambd * R @ mean_corresponding

    return lambd, R, T