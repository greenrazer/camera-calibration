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
    # mirrored x = -x y = y z = z
    real_points_m = np.array([
        [0,0,0], # White
        [DIST,0,0], # Purple
        [-DIST,0,0], # Silver
        [0,DIST,0], # Orange
        [0,-DIST,0], # Red
        [0,0,DIST], # Blue
        [0,0,-DIST], # Yellow
    ])

    # real_points_m = np.random.rand(*real_points_m.shape)
    # points = np.random.rand(real_points_m.shape[0], 2)

    colors = [
        [255,255,255], # White
        [250, 150, 150], # Purple
        [220, 220, 220], # Silver
        [0, 165, 255],  # Orange
        [0, 0, 255], # Red
        [255, 0, 0], # Blue
        [0, 255, 255], # Yellow
    ]

    imgs = [
        cv2.imread("photos/star.jpg"),
        cv2.imread("photos/star2.jpg"),
        cv2.imread("photos/star3.jpg"),
        # cv2.imread("photos/star4.jpg"),
        cv2.imread("photos/star5.jpg"),
        cv2.imread("photos/star6.jpg"),
    ]


    # fuck_you = []
    # for num in range(len(imgs)):
    #     points = image_utils.show_image_select_points('Select points', imgs[num], width=800, colors=colors, uv=True)
    #     fuck_you.append(f"points{num} = {points}")

    # for fuck in fuck_you:
    #     print(fuck)

    # return
    # im = image_utils.draw_points_on_image(imgs[0], points, colors)
    # cv2.imshow("as", im)
    # cv2.waitKey(0)

    points0 = [(0.4766666666666667, 0.4425), (0.8416666666666667, 0.7925), (0.20333333333333334, 0.2175), (0.62, 0.56125), (0.20333333333333334, 0.12375), (0.03, 0.69625), (0.9183333333333333, 0.17875)]
    points1 = [(0.51, 0.5), (0.86125, 0.58), (0.20625, 0.4125), (0.4925, 0.25125), (0.515, 0.645), (0.59125, 0.25), (0.40375, 0.83375)]
    points2 = [(0.49375, 0.59375), (0.4725, 0.69625), (0.53, 0.42125), (0.75, 0.54625), (0.2125, 0.65875), (0.4375, 0.37875), (0.58125, 0.87125)]
    # points3 = [(0.51625, 0.58875), (0.50875, 0.515), (0.53125, 0.67625), (0.64, 0.58), (0.395, 0.605), (0.51125, 0.4925), (0.5225, 0.63625)] 
    points4 = [(0.53375, 0.56875), (0.54125, 0.6625), (0.5225, 0.455), (0.66125, 0.5275), (0.39875, 0.615), (0.48875, 0.485), (0.59125, 0.66625)]
    points5 = [(0.615, 0.33875), (0.68, 0.48625), (0.53375, 0.165), (0.72875, 0.2375), (0.4925, 0.45375), (0.5, 0.33625), (0.77375, 0.34125)]

    # away    wrong    neg neg
    # towards wrong    pos neg
    # towards right    neg pos
    # away    right    pos pos
    # towards wrong    pos neg
    # towards wrong    pos neg

    # x y z
    # flip ok flip
    # 


    all_points = [points0, points1, points2, points4, points5]

    # print(points)
    # points = np.array(points)

    # mpf = draw_utils.MultiPartFig(2,5)

    # for points, im in zip(all_points, imgs):
    #     position, rotation, internal = li_utils.calibrate_camera(real_points_m, points)
    #     mpf.add_3d_part(real_points_m, position, rotation, internal)
    #     mpf.add_image(im)

    # mpf.show()
    # mpf.main_show()


    # np.set_printoptions(precision=1)
    # np.set_printoptions(suppress=True)

    num = 3
    for num in range(len(imgs)):

        points = np.array(all_points[num])
        # points[:,1] = 1 - points[:,1]
        imag = imgs[num]

        # drawn = image_utils.draw_points_on_image(imag.copy(), points, radius=5, colors=colors)
        # image_utils.show_image("howdy", drawn)

        def display(p):
            internal, rotation, position = li_utils.get_projection_product_matricies(p)

            print("internal")
            print(internal)

            # print("KR")
            # print(internal @ rotation)

            # lt = draw_utils.show_scene(real_points_m, internal, rotation, position)

            # li_utils.remove_negitive_focal_inplace(internal, rotation)

            # print("KR")
            # print(internal @ rotation)

            # print("I")
            # print(internal)
            # print("R")
            # print(rotation)
            # print("P")
            # print(position)
            plt = draw_utils.show_scene(real_points_m, internal, rotation, position)
        
        def draw_on_pic(P, booll=False):
            new_points = li_utils.camera_project_points(P, real_points_m)

            drawn = image_utils.draw_points_on_image(imag, new_points, radius=20, colors=colors)
            # if booll:
            #     image_utils.show_image("howdy", drawn)
            print(new_points)
            display(P)


        # P = li_utils.dlt(real_points_m, points)

        # print(P)

        # display(P)
        # draw_on_pic(P, True)


        # P = li_utils.camera_projection_levenberg_marquardt(real_points_m, points, P, callback = draw_on_pic, call_every=1, iters=100)

        # P = li_utils.camera_projection_levenberg_marquardt(real_points_m, points, P, callback = None)

        # display(P)

        # internal, rotation, position = li_utils.calibrate_camera(real_points_m, np.array(points1))
        # print(position)
        # plt = draw_utils.show_scene(real_points_m, internal, rotation, position)

        # P = li_utils.product_matricies_to_projection_matrix(internal, rotation, position)
        
        # li_utils.test_decomp2(real_points_m, np.array(points1))
        # # w = li_utils.SO_to_SE(rotation)
        # # R = li_utils.SE_to_SO(w)
        P = li_utils.calibrate_camera(real_points_m, points)
        draw_on_pic(P, booll=True)
        # # li_utils.test_numerical_jacobian(real_points_m, np.array(points2))
        # # li_utils.test_numerical_jacobian(real_points_m, np.array(points3))
        # # li_utils.test_numerical_jacobian(real_points_m, np.array(points4))
        # # li_utils.test_numerical_jacobian(real_points_m, np.array(points5))
        # drawn = image_utils.draw_points_on_image(imgs[0], new_points, colors)
        # image_utils.show_image("howdy", drawn)

if __name__ == '__main__':
    default()



