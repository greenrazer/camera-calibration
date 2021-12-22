import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_vec(ax, from_pt, to_pt, color):
    ax.quiver([from_pt[0]], [from_pt[1]], [from_pt[2]], [to_pt[0]], [to_pt[1]], [to_pt[2]], color=[color])

def draw_point(ax, point, color):
    ax.scatter([point[0]], [point[1]], [point[2]], color=[color])

def show_scene(points, inter, rot, pos, scale=1, box_radius=2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    create_3d_scene(ax, points, pos, rot, inter, scale=scale, box_radius=box_radius)
    plt.show()

def draw_rotation_matrix(ax, rot, at=[0,0,0], scale=1):
    draw_vec(ax, at, rot[0]*scale, [1,0,0])
    draw_vec(ax, at, rot[1]*scale, [0,1,0])
    draw_vec(ax, at, rot[2]*scale, [0,0,1])

def create_3d_scene(ax, points, pos, rot, inter, scale=1, box_radius=2):

    # draw_vec(ax, pos, rot[0]*scale, [1,0,0])
    # draw_vec(ax, pos, rot[1]*scale, [0,1,0])
    # draw_vec(ax, pos, rot[2]*scale, [0,0,1])
    draw_rotation_matrix(ax, rot, pos)
    draw_point(ax, pos, [0,0,0])

    colors = [
        [1,1,1], # White
        [1, 1, 0], # Yellow
        [0, 0, 1], # Blue
        [0.4, 0.4, 1], # Purple
        [0.8, 0.8, 0.8], # Silver
        [1, 0, 0], # Red
        [1, 0.6, 0], # Orange
    ]

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

