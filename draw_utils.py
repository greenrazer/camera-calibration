import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE_COLORS = [
        [1,1,1], # White
        [0.4, 0.4, 1], # Purple
        [0.8, 0.8, 0.8], # Silver
        [1, 0.6, 0], # Orange
        [1, 0, 0], # Red
        [0, 0, 1], # Blue
        [1, 1, 0], # Yellow
    ]

def draw_vec(ax, from_pt, to_pt, color):
    ax.quiver([from_pt[0]], [from_pt[1]], [from_pt[2]], [to_pt[0]], [to_pt[1]], [to_pt[2]], color=[color])

def draw_point(ax, point, color):
    ax.scatter([point[0]], [point[1]], [point[2]], color=[color])

def show_scene(points, inter, rot, pos, colors= [1, 1, 1], scale=1, box_radius=2, line=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if line is not None:
        ax.plot(line[0], line[1], line[2])

    create_3d_scene(ax, points, pos, rot, inter, scale=scale, box_radius=box_radius)
    plt.show()

def show_multi_cam_scene(points, positions, rotations, colors=BASE_COLORS):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for p, R in zip(positions, rotations):
            draw_rotation_matrix(ax, R.T, p)
            draw_point(ax, p, [0,0,0])

    i = 0
    for pt in points:
        draw_point(ax, pt, colors[i%len(colors)])
        i+=1

    box_radius = 2
    ax.set_xlim([-box_radius,box_radius])
    ax.set_ylim([-box_radius, box_radius])
    ax.set_zlim([-box_radius, box_radius])

    plt.show()


def draw_rotation_matrix(ax, rot, at=[0,0,0], scale=1):
    # negitive y because the image is upsidedown in the computer
    # like y=0 is the top y >> 0 is the bottom
    # x works as expected, however since flipping y changes left and right
    r = rot.T
    draw_vec(ax, at, r[:,0]*scale, [1,0,0]) # red y up - x
    draw_vec(ax, at, r[:,1]*scale, [0,1,0]) # green x right - y
    draw_vec(ax, at, r[:,2]*scale, [0,0,1]) # blue z forward - z


def create_3d_scene(ax, points, pos, rot, inter, colors=BASE_COLORS, scale=1, box_radius=2):

    draw_rotation_matrix(ax, rot, pos)
    draw_point(ax, pos, [0,0,0])

    i = 0
    for pt in points:
        draw_point(ax, pt, colors[i%len(colors)])
        i+=1

    ax.set_xlim([-box_radius,box_radius])
    ax.set_ylim([-box_radius, box_radius])
    ax.set_zlim([-box_radius, box_radius])

class MultiPartFig:
    def __init__(self, n_rows, n_columns):
        self.rows = n_rows
        self.columns = n_columns

        self.curr = 1

        self.fig = plt.figure()
    
    def add_3d_part(self, points, pos, rot, inter, scale=1, box_radius=2):
        ax = self.fig.add_subplot(self.rows, self.columns, self.curr, projection='3d')
        create_3d_scene(ax, points, pos, rot, inter, scale, box_radius)
        self.curr += 1

    def add_image(self, image):
        ax = self.fig.add_subplot(self.rows, self.columns, self.curr + self.columns -1)
        ax.imshow(image)

    def show(self):
        self.fig.show()

    def main_show(self):
        plt.show()

