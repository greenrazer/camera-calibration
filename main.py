import cv2
import numpy as np

import image_utils
import utils
import li_utils
import draw_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def try1():
    train_card = image_utils.open_and_white_balance("photos/test_card.jpg")
    
    train =  image_utils.show_img_select_rois('Select Color', train_card, single=True)
    hsv_train = cv2.cvtColor(train, cv2.COLOR_BGR2HSV)
    lower, upper = image_utils.hsv_color_filter(hsv_train)

    target = image_utils.open_and_white_balance("photos/star.jpg", hsv=True)

    inv_mask = image_utils.color_filter_mask(target, lower, upper, inv=True)

    image_utils.show_image('howdi', inv_mask)
    #colors too close to one another.

    keypoints = image_utils.detect_circular_blobs(inv_mask)

    keypoint_img = cv2.drawKeypoints(target, keypoints, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    image_utils.show_image('howdi', cv2.cvtColor(keypoint_img, cv2.COLOR_HSV2BGR))

def try2():
    target = cv2.imread("photos/star.jpg")
    points = image_utils.show_image_select_points('Select points', target)

    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    target_patch = image_utils.extract_square_patch(target, points[0], 150)
    target_patch_hsv = cv2.cvtColor(target_patch, cv2.COLOR_BGR2HSV)
    cv2.imshow('fuck', target_patch)
    
    print((points[0][0], points[0][1]))
    print(target_hsv.shape)
    point_color = target_hsv[points[0][0], points[0][1] , :]
    lower_c, upper_c = image_utils.color_hsv_epsilon(point_color, 20, 100, 255)
    mask = image_utils.color_filter_mask(target_patch_hsv, lower_c, upper_c)
    
    sw = utils.SliderWindow('square_patch')
    sw.add_image(mask)
    sw.add_slider('h_epsilon', 20)
    sw.add_slider('s_epsilon', 100)
    sw.add_slider('v_epsilon', 255)

    def on_change(obj):
        eh = obj.get_slider_value('h_epsilon')
        es = obj.get_slider_value('s_epsilon')
        ev = obj.get_slider_value('v_epsilon')

        lower_c, upper_c = image_utils.color_hsv_epsilon(point_color, eh, es, ev)
        mask = image_utils.color_filter_mask(target_patch_hsv, lower_c, upper_c)
        sw.display_img(mask)
    
    sw.add_callback(on_change)

    sw.show()

    if cv2.waitKey(0):
        cv2.destroyAllWindows()

def try3():
    vs = cv2.VideoCapture(0)

    obj_track = image_utils.ObjectTracker(draw_debug=True)

    while True:
        ret, frame = vs.read()
        
        boxes = obj_track.update(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            obj_track.add_new_tracker(frame)
        if key == ord('q'):
            break        

        cv2.imshow('hi', frame)

    cv2.destroyAllWindows()

def default():
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

    # real_points_m = np.random.rand(*real_points_m.shape)
    # points = np.random.rand(real_points_m.shape[0], 2)

    colors = [
        [255,255,255], # White
        [0, 255, 255], # Yellow
        [255, 0, 0], # Blue
        [250, 230, 230], # Purple
        [220, 220, 220], # Silver
        [0, 0, 255], # Red
        [0, 165, 255], # Orange
    ]

    imgs = [
        cv2.imread("photos/star.jpg"),
        cv2.imread("photos/star2.jpg"),
        cv2.imread("photos/star3.jpg"),
        cv2.imread("photos/star4.jpg"),
        cv2.imread("photos/star5.jpg"),
    ]

    # points = image_utils.show_image_select_points('Select points', imgs[0], width=800, colors=colors, uv=True)
    points1 = [(0.515, 0.4975), (0.40375, 0.835), (0.59125, 0.25), (0.86375, 0.57875), (0.2075, 0.41625), (0.52, 0.64), (0.49625, 0.2525)]
    points2 = [[0.47166667,0.4475],[0.925,0.18], [0.03333333 ,0.69625], [0.83833333 ,0.79], [0.20833333 ,0.21625], [0.21333333 ,0.12], [0.62,0.56]]
    points3 = [(0.495, 0.59), (0.58, 0.87125), (0.43625, 0.37875), (0.47625, 0.69375), (0.5275, 0.42), (0.2125, 0.6575), (0.7525, 0.545)]
    points4 = [(0.51625, 0.59125), (0.525, 0.655), (0.50875, 0.4925), (0.51, 0.515), (0.53125, 0.67625), (0.39625, 0.605), (0.64, 0.58125)]
    points5 = [(0.53375, 0.5675), (0.5925, 0.665), (0.49, 0.48625), (0.5425, 0.66125), (0.52125, 0.45375), (0.4, 0.615), (0.6625, 0.52625)]

    all_points = np.array([points1, points2, points3, points4, points5])

    # print(points)
    # points = np.array(points)

    # mpf = draw_utils.MultiPartFig(2,5)

    # for points, im in zip(all_points, imgs):
    #     position, rotation, internal = li_utils.calibrate_camera(real_points_m, points)
    #     mpf.add_3d_part(real_points_m, position, rotation, internal)
    #     mpf.add_image(im)

    # mpf.show()
    # mpf.main_show()


    np.set_printoptions(precision=1)
    np.set_printoptions(suppress=True)

    def display(p):
        internal, rotation, position = li_utils.get_projection_product_matricies(p)
        print(position)
        plt = draw_utils.show_scene(real_points_m, internal, rotation, position)

    points = np.array(points2)

    P = li_utils.dlt(real_points_m, points)

    display(P)

    # P = li_utils.camera_projection_levenberg_marquardt(real_points_m, points, P, callback = None)

    # display(P)

    # internal, rotation, position = li_utils.calibrate_camera(real_points_m, np.array(points1))
    # print(position)
    # plt = draw_utils.show_scene(real_points_m, internal, rotation, position)

    # P = li_utils.product_matricies_to_projection_matrix(internal, rotation, position)
    # new_points = li_utils.camera_project_points(P, real_points_m)
    # print(new_points)

    # drawn = image_utils.draw_points_on_image(imgs[0], new_points, colors)
    # image_utils.show_image("howdy", drawn)
    
    # li_utils.test_decomp2(real_points_m, np.array(points1))
    # # w = li_utils.SO_to_SE(rotation)
    # # R = li_utils.SE_to_SO(w)
    new_points = li_utils.test_numerical_jacobian(real_points_m, np.array(points1))
    # # li_utils.test_numerical_jacobian(real_points_m, np.array(points2))
    # # li_utils.test_numerical_jacobian(real_points_m, np.array(points3))
    # # li_utils.test_numerical_jacobian(real_points_m, np.array(points4))
    # # li_utils.test_numerical_jacobian(real_points_m, np.array(points5))
    # drawn = image_utils.draw_points_on_image(imgs[0], new_points, colors)
    # image_utils.show_image("howdy", drawn)

if __name__ == '__main__':
    default()



